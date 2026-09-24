"""Бэкенды генерации для Qwen-Image-2.1.

- `DiffusersBackend` — настоящая модель через `diffusers.QwenImage21Pipeline` (нужна CUDA).
- `MockBackend` — заглушка без модели: рисует картинку-плейсхолдер с параметрами запроса.
  Нужна, чтобы отлаживать интерфейс на машине без подходящей видеокарты.

Выбор делает `load_backend()`: `QWEN_MOCK=1` принудительно включает заглушку,
иначе берётся настоящая модель, если доступна CUDA.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen-Image-2.1")

# Рекомендованные размеры из README модели (родное разрешение 2K).
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

MAX_REFERENCE_IMAGES = 10

# Пайплайн округляет стороны вниз до кратного vae_scale_factor * 2.
SIZE_MULTIPLE = 32

ProgressFn = Callable[[int, int], None]


def size_for_ratio(ratio: str, resolution: int) -> tuple[int, int]:
    """Размер для соотношения сторон. На 2048 — официальная таблица, иначе она же, масштабированная."""
    w, h = ASPECT_RATIOS[ratio]
    if resolution == 2048:
        return w, h
    scale = resolution / 2048
    return (
        max(SIZE_MULTIPLE, round(w * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE),
        max(SIZE_MULTIPLE, round(h * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE),
    )


def size_like_image(image: Image.Image, resolution: int) -> tuple[int, int]:
    """Размер с пропорциями `image` и площадью resolution², как `calculate_dimensions` в пайплайне."""
    ratio = image.width / image.height
    width = math.sqrt(resolution * resolution * ratio)
    height = width / ratio
    return round(width / SIZE_MULTIPLE) * SIZE_MULTIPLE, round(height / SIZE_MULTIPLE) * SIZE_MULTIPLE


@dataclass
class GenRequest:
    prompt: str
    images: list[Image.Image] = field(default_factory=list)
    # Если не заданы — пайплайн сам берёт пропорции последнего референса (или квадрат) с площадью output_resolution².
    width: int | None = None
    height: int | None = None
    steps: int = 40
    seed: int = 42
    negative_prompt: str = ""
    true_cfg_scale: float = 1.0
    # Площадь, к которой приводятся референсы (и вывод, если размер не задан).
    output_resolution: int = 1024
    use_kv_cache: bool = True
    num_images: int = 1

    def resolved_size(self) -> tuple[int, int]:
        """Итоговый размер вывода — повторяет логику `QwenImage21Pipeline.__call__`."""
        width, height = self.width, self.height
        if self.images:
            calc_w, calc_h = size_like_image(self.images[-1], self.output_resolution)
            width, height = width or calc_w, height or calc_h
        width = width or self.output_resolution
        height = height or self.output_resolution
        return width // SIZE_MULTIPLE * SIZE_MULTIPLE, height // SIZE_MULTIPLE * SIZE_MULTIPLE


def drop_opaque_alpha(image: Image.Image) -> Image.Image:
    """VAE модели всегда декодирует 4 канала. Если альфа везде непрозрачна — отдаём обычный RGB."""
    if image.mode == "RGBA" and image.getchannel("A").getextrema()[0] == 255:
        return image.convert("RGB")
    return image


class DiffusersBackend:
    is_mock = False

    def __init__(self) -> None:
        import torch
        from diffusers import QwenImage21Pipeline

        self._torch = torch
        pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        offload = os.environ.get("QWEN_OFFLOAD", "auto")
        if offload == "auto":
            # bf16-веса DiT + энкодера + VAE ≈ 33 GB, плюс активации на 2K и референсы.
            offload = "none" if vram_gb >= 60 else "model"
        if offload == "model":
            pipe.enable_model_cpu_offload()
        elif offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")

        compiled = os.environ.get("QWEN_COMPILE") == "1"
        if compiled:
            from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor

            # Flex attention без compile материализует матрицу внимания в fp32 — только вместе.
            pipe.transformer.set_attn_processor(QwenImage21FlexAttnProcessor())
            pipe.transformer.compile()

        self.pipe = pipe
        # Один GPU — одна генерация за раз; очередь держит Gradio.
        self._lock = threading.Lock()
        gpu = torch.cuda.get_device_name(0)
        self.description = (
            f"{MODEL_ID} · {gpu} ({vram_gb:.0f} GB) · offload: {offload}"
            + (" · flex+compile" if compiled else "")
        )

    def generate(self, req: GenRequest, progress: ProgressFn | None = None) -> list[Image.Image]:
        torch = self._torch
        kwargs = dict(
            prompt=req.prompt,
            num_inference_steps=req.steps,
            generator=torch.Generator("cuda").manual_seed(req.seed),
            output_resolution=req.output_resolution,
            use_kv_cache=req.use_kv_cache,
            num_images_per_prompt=req.num_images,
        )
        if req.images:
            kwargs["image"] = req.images
        if req.width and req.height:
            kwargs.update(width=req.width, height=req.height)
        if req.negative_prompt.strip() and req.true_cfg_scale > 1:
            kwargs.update(negative_prompt=req.negative_prompt, true_cfg_scale=req.true_cfg_scale)
        if progress:

            def on_step_end(pipe, step, timestep, callback_kwargs):
                progress(step + 1, req.steps)
                return callback_kwargs

            kwargs["callback_on_step_end"] = on_step_end

        with self._lock, torch.inference_mode():
            images = self.pipe(**kwargs).images
        return [drop_opaque_alpha(im) for im in images]


class MockBackend:
    """Имитирует генерацию: картинка нужного размера с текстом запроса и миниатюрами референсов."""

    is_mock = True
    description = "ЗАГЛУШКА — модель не загружена, картинки ненастоящие"

    # Реальные 2K-плейсхолдеры бессмысленно гонять через браузер — рисуем уменьшенные с теми же пропорциями.
    PREVIEW_MAX_SIDE = 768

    def generate(self, req: GenRequest, progress: ProgressFn | None = None) -> list[Image.Image]:
        from prompts import RGBA_PREFIX

        width, height = req.resolved_size()
        scale = min(1.0, self.PREVIEW_MAX_SIDE / max(width, height))
        pw, ph = max(64, int(width * scale)), max(64, int(height * scale))
        transparent = RGBA_PREFIX.lower() in req.prompt.lower()

        for step in range(req.steps):
            time.sleep(0.02)
            if progress:
                progress(step + 1, req.steps)

        results = []
        for n in range(req.num_images):
            hue = (req.seed * 37 + n * 61) % 255
            if transparent:
                canvas = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
                ImageDraw.Draw(canvas).ellipse(
                    (pw * 0.2, ph * 0.2, pw * 0.8, ph * 0.8), fill=(hue, 120, 255 - hue, 255)
                )
            else:
                canvas = Image.new("RGB", (pw, ph), (hue, 90, 255 - hue))

            thumb_side = max(32, min(pw, ph) // 6)
            for i, ref in enumerate(req.images):
                thumb = ref.convert("RGBA").copy()
                thumb.thumbnail((thumb_side, thumb_side))
                canvas.paste(thumb, (8 + i * (thumb_side + 4), ph - thumb_side - 8), thumb)

            lines = [
                f"MOCK #{n + 1}  {width}x{height}  seed={req.seed}  steps={req.steps}",
                f"refs={len(req.images)}  res={req.output_resolution}  cfg={req.true_cfg_scale}"
                f"  kv_cache={req.use_kv_cache}",
                "",
                *_wrap(req.prompt, max(20, pw // 8)),
            ]
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default(size=max(11, pw // 50))
            draw.multiline_text((10, 10), "\n".join(lines[:20]), fill=(255, 255, 255, 255), font=font,
                                stroke_width=2, stroke_fill=(0, 0, 0, 255))
            results.append(canvas)
        return results


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


def load_backend() -> DiffusersBackend | MockBackend:
    if os.environ.get("QWEN_MOCK") == "1":
        return MockBackend()
    try:
        import torch
    except ImportError:
        print("[backend] torch не установлен — работаю в режиме заглушки")
        return MockBackend()
    if not torch.cuda.is_available():
        print("[backend] CUDA недоступна — работаю в режиме заглушки (QWEN_MOCK=1 делает это явно)")
        return MockBackend()
    return DiffusersBackend()
