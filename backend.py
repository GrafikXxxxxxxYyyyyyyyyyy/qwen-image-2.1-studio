"""Бэкенды генерации для Qwen-Image-2.1.

- `DiffusersBackend` — настоящая модель через `diffusers.QwenImage21Pipeline` (нужна CUDA).
- `MockBackend` — заглушка без модели: рисует картинку-плейсхолдер с параметрами запроса.
  Нужна, чтобы отлаживать интерфейс на машине без подходящей видеокарты.

Выбор делает `load_backend()`: `QWEN_MOCK=1` принудительно включает заглушку,
иначе берётся настоящая модель, если доступна CUDA.
"""

from __future__ import annotations

import gc
import math
import os
import re
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

# Пик памяти на 2K — декодирование VAE (+~26 GB); без expandable_segments фрагментация роняет его в OOM на 48 GB.
# Действует, только если выставлено до первой аллокации CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen-Image-2.1")

# Трансформер по умолчанию — GGUF-квант uncensored-версии: Q8_0 ≈ 7 GB вместо 14 GB в bf16, чтобы вместе с
# turbo-LoRA и 2K-генерацией уложиться в 24 GB (RTX 3090/4090). Энкодер, VAE и конфиги берутся из MODEL_ID.
# https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF (лицензия Qwen Research, как у модели).
# QWEN_GGUF=none — оригинальный bf16-трансформер из MODEL_ID.
GGUF_REPO = os.environ.get("QWEN_GGUF_REPO", "abenzerps/Qwen-Image-2.1-Uncensored-GGUF")
GGUF_FILE = os.environ.get("QWEN_GGUF", "qwen-image-2.1-UC-Q8_0.gguf")
if GGUF_FILE.lower() in ("", "0", "none"):
    GGUF_FILE = ""

# Turbo: DMD-дистилляция от Viggle, LoRA поверх базового трансформера. 6 шагов по фиксированным sigma, без CFG.
# https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo (лицензия Qwen Research, как у модели).
TURBO_REPO = os.environ.get("QWEN_TURBO_REPO", "Viggle/Qwen-Image-2.1-viggle-turbo")
TURBO_WEIGHTS = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
TURBO_ENABLED = os.environ.get("QWEN_TURBO", "1") == "1"
TURBO_SIGMAS = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
TURBO_STEPS = len(TURBO_SIGMAS)

MAX_REFERENCE_IMAGES = 10

# Пайплайн округляет стороны вниз до кратного vae_scale_factor * 2.
SIZE_MULTIPLE = 32
# Пределы стороны вывода. Модель обучена на площадях 1024²–2048² (до 2752 по длинной стороне);
# за этими пределами качество падает, а на 48 GB растёт риск OOM.
MIN_SIDE = 256
MAX_SIDE = 3072

ProgressFn = Callable[[int, int], None]


def clamp_size(width: float, height: float) -> tuple[int, int]:
    """Сторона — кратное SIZE_MULTIPLE в пределах [MIN_SIDE, MAX_SIDE]."""
    snap = lambda v: min(MAX_SIDE, max(MIN_SIDE, round(v / SIZE_MULTIPLE) * SIZE_MULTIPLE))  # noqa: E731
    return snap(width), snap(height)


def size_for_aspect(aspect: float, resolution: int) -> tuple[int, int]:
    """Размер с отношением ширины к высоте `aspect` и площадью resolution², как `calculate_dimensions` в пайплайне."""
    width = math.sqrt(resolution * resolution * aspect)
    return clamp_size(width, width / aspect)


def size_like_image(image: Image.Image, resolution: int) -> tuple[int, int]:
    return size_for_aspect(image.width / image.height, resolution)


def parse_ratio(text: str) -> float | None:
    """'16:9' → 1.777…; None, если строка не похожа на соотношение сторон (так отвечает переписчик промптов)."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:x×/]\s*(\d+(?:\.\d+)?)\s*", text or "")
    if not m or float(m.group(2)) == 0:
        return None
    return float(m.group(1)) / float(m.group(2))


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
    # Turbo-LoRA: steps должно быть TURBO_STEPS, CFG не применяется.
    turbo: bool = False

    def resolved_size(self) -> tuple[int, int]:
        """Итоговый размер вывода — повторяет логику `QwenImage21Pipeline.__call__`."""
        width, height = self.width, self.height
        if self.images:
            calc_w, calc_h = size_like_image(self.images[-1], self.output_resolution)
            width, height = width or calc_w, height or calc_h
        width = width or self.output_resolution
        height = height or self.output_resolution
        return width // SIZE_MULTIPLE * SIZE_MULTIPLE, height // SIZE_MULTIPLE * SIZE_MULTIPLE


# На обычных (не RGBA) генерациях альфа не ровно 255, а шумит: в основном 237–255, единичные пиксели до ~190.
# У настоящих RGBA прозрачна заметная часть кадра (у стикера на прозрачном фоне — треть).
TRANSPARENT_SHARE_MIN = 0.001


def drop_opaque_alpha(image: Image.Image) -> Image.Image:
    """VAE модели всегда декодирует 4 канала. Если прозрачных пикселей (альфа < 128) почти нет — отдаём RGB."""
    if image.mode != "RGBA":
        return image
    hist = image.getchannel("A").histogram()
    if sum(hist[:128]) < TRANSPARENT_SHARE_MIN * sum(hist):
        return image.convert("RGB")
    return image


# Выше этого (ширина × высота × число картинок) на картах < 40 GB VAE сразу декодирует тайлами.
# 1024² без тайлинга на 24 GB проходит (пик ~16 GB), 2048² — нет.
TILE_UPFRONT_PIXELS = 1280 * 1280


def _load_gguf_transformer(model_id: str, repo: str, filename: str):
    import torch
    from diffusers import GGUFQuantizationConfig, QwenImage21Transformer2DModel
    from diffusers.quantizers.gguf.utils import GGUFLinear, dequantize_gguf_tensor
    from huggingface_hub import hf_hub_download

    transformer = QwenImage21Transformer2DModel.from_single_file(
        hf_hub_download(repo, filename),
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    # stable-diffusion.cpp хранит мелкие тензоры (нормы, txt_in) в BF16, а diffusers грузит BF16 из GGUF как сырые
    # байты и распаковывает их только в GGUFLinear. Остальные модули (RMSNorm) получили бы uint8 двойной длины.
    for module in transformer.modules():
        if isinstance(module, GGUFLinear):
            continue
        for name, param in list(module.named_parameters(recurse=False)):
            if hasattr(param, "quant_type"):
                unpacked = dequantize_gguf_tensor(param).to(torch.bfloat16)
                setattr(module, name, torch.nn.Parameter(unpacked, requires_grad=False))
    return transformer


class DiffusersBackend:
    is_mock = False

    def __init__(self) -> None:
        import torch
        from diffusers import QwenImage21Pipeline

        self._torch = torch
        # Переданный трансформер from_pretrained не скачивает — из MODEL_ID идут только энкодер, VAE и конфиги.
        extra = {"transformer": _load_gguf_transformer(MODEL_ID, GGUF_REPO, GGUF_FILE)} if GGUF_FILE else {}
        pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16, **extra)

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        self._vram_gb = vram_gb
        offload = os.environ.get("QWEN_OFFLOAD", "auto")
        if offload == "auto":
            # Веса: энкодер ≈ 17 GB, трансформер ≈ 14 GB (GGUF Q8_0 ≈ 7 GB), turbo-LoRA ≈ 1.3 GB, VAE < 1 GB.
            # Денойзинг добавляет к весам до ~4 GB на 2K, декодирование VAE на 2K — ещё ~26 GB (с тайлингом — ~2 GB).
            offload = "none" if vram_gb >= 64 else "encoder" if vram_gb >= (20 if GGUF_FILE else 40) else "model"
        if offload == "model":
            # Гоняет и трансформер через PCIe на каждой генерации: на A6000 +40 с к 1024².
            pipe.enable_model_cpu_offload()
        elif offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload == "encoder":
            # Энкодер нужен только в начале генерации — держим его в RAM и подгружаем послойно (+1–3 с).
            from diffusers.hooks import apply_group_offloading

            pipe.transformer.to("cuda")
            pipe.vae.to("cuda")
            apply_group_offloading(pipe.text_encoder, onload_device=torch.device("cuda"),
                                   offload_device=torch.device("cpu"), offload_type="leaf_level", use_stream=True)
        else:
            pipe.to("cuda")

        self.has_turbo = False
        if TURBO_ENABLED:
            from diffusers import FlowMatchEulerDiscreteScheduler

            # LoRA не сливаем с весами: слияние в bf16 заметно меняет картинку, а выключенная LoRA почти бесплатна.
            pipe.load_lora_weights(TURBO_REPO, weight_name=TURBO_WEIGHTS)
            pipe.disable_lora()
            # У базового планировщика shift_terminal=0.02 — для turbo он портит последний шаг.
            self._schedulers = {False: pipe.scheduler,
                                True: FlowMatchEulerDiscreteScheduler.from_pretrained(TURBO_REPO, subfolder="scheduler")}
            self.has_turbo = True

        compiled = os.environ.get("QWEN_COMPILE") == "1"
        if compiled:
            from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor

            # Flex attention без compile материализует матрицу внимания в fp32 — только вместе.
            pipe.transformer.set_attn_processor(QwenImage21FlexAttnProcessor())
            pipe.transformer.compile()

        if offload == "encoder":
            # С use_stream порядок слоёв записывается на первом проходе, и дальше каждый слой подгружает
            # следующий вместо себя. Если первой была генерация с картинками, в цепочку попадает vision-энкодер,
            # которого нет в text-to-image: цепочка рвётся, и rotary-слой (только буферы, без параметров —
            # запасная синхронная подгрузка diffusers его не видит) остаётся на CPU: «tensors on different
            # devices». Прогон без картинок записывает только общие слои, а слои vision-энкодера остаются вне
            # цепочки и подгружают себя сами. Без use_stream генерация медленнее на ~10 с.
            # Строго после load_lora_weights: она заново вешает хуки offloading и сбрасывает записанный порядок.
            with torch.inference_mode():
                pipe.encode_prompt("warmup", device=torch.device("cuda"))

        self.pipe = pipe
        # Один GPU — одна генерация за раз; очередь держит Gradio.
        self._lock = threading.Lock()
        gpu = torch.cuda.get_device_name(0)
        self.description = (
            f"{f'{GGUF_REPO}/{GGUF_FILE}' if GGUF_FILE else MODEL_ID} · {gpu} ({vram_gb:.0f} GB) · offload: {offload}"
            + (" · turbo LoRA" if self.has_turbo else "")
            + (" · flex+compile" if compiled else "")
        )

    def generate(self, req: GenRequest, progress: ProgressFn | None = None) -> list[Image.Image]:
        torch = self._torch
        turbo = req.turbo and self.has_turbo
        kwargs = dict(
            prompt=req.prompt,
            num_inference_steps=req.steps,
            output_resolution=req.output_resolution,
            use_kv_cache=req.use_kv_cache,
            num_images_per_prompt=req.num_images,
        )
        if req.images:
            kwargs["image"] = req.images
        if req.width and req.height:
            kwargs.update(width=req.width, height=req.height)
        if turbo:
            kwargs["sigmas"] = TURBO_SIGMAS
        elif req.negative_prompt.strip() and req.true_cfg_scale > 1:
            kwargs.update(negative_prompt=req.negative_prompt, true_cfg_scale=req.true_cfg_scale)
        if progress:

            def on_step_end(pipe, step, timestep, callback_kwargs):
                progress(step + 1, req.steps)
                return callback_kwargs

            kwargs["callback_on_step_end"] = on_step_end

        # Декодирование VAE без тайлинга — ~6 GB на мегапиксель вывода. На картах меньше 40 GB крупный вывод
        # сразу декодируем тайлами: иначе OOM случится после всего денойзинга, и генерация пойдёт заново.
        width, height = req.resolved_size()
        tile_upfront = self._vram_gb < 40 and width * height * req.num_images > TILE_UPFRONT_PIXELS

        with self._lock, torch.inference_mode():
            if self.has_turbo:
                (self.pipe.enable_lora if turbo else self.pipe.disable_lora)()
                self.pipe.scheduler = self._schedulers[turbo]
            images = None
            if not tile_upfront:
                try:
                    images = self._run(kwargs, req.seed)
                except torch.OutOfMemoryError as e:
                    self._release(e)
            if images is None:
                # Обычно OOM — это декодирование VAE (большой размер, несколько картинок). С тайлингом оно
                # укладывается в память, но медленнее на ~10 с — поэтому на больших картах только как повтор.
                self.pipe.vae.enable_tiling()
                try:
                    images = self._run(kwargs, req.seed)
                except BaseException as e:
                    self._release(e)
                    raise
                finally:
                    self.pipe.vae.disable_tiling()
        return [drop_opaque_alpha(im) for im in images]

    def _release(self, exc: BaseException) -> None:
        """Освобождает видеопамять упавшей генерации.

        Traceback исключения держит фреймы пайплайна со всеми тензорами (KV-кэш, латенты, эмбеддинги),
        а само исключение живёт дольше генерации: app.py заворачивает его в gr.Error (`from e`), а Gradio
        хранит ошибку. Без очистки фреймов после первого OOM в VRAM застревают десятки GB и падают
        все следующие генерации.
        """
        traceback.clear_frames(exc.__traceback__)
        gc.collect()
        self._torch.cuda.empty_cache()

    def _run(self, kwargs: dict, seed: int) -> list[Image.Image]:
        # Генератор каждый раз свежий: повтор после OOM должен дать ту же картинку.
        return self.pipe(**kwargs, generator=self._torch.Generator("cuda").manual_seed(seed)).images


class MockBackend:
    """Имитирует генерацию: картинка нужного размера с текстом запроса и миниатюрами референсов."""

    is_mock = True
    has_turbo = True
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
                f"  kv_cache={req.use_kv_cache}  turbo={req.turbo}",
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
