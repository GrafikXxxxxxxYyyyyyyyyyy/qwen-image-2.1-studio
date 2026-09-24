"""Переписывание промптов официальными моделями Qwen-Image-2.1-PE-T2I / PE-I2I.

Два способа запуска:
- vLLM-сервер (рекомендуется): `prompt_rewrite/serve.sh` из репозитория Qwen, адрес в
  `QWEN_PE_T2I_URL` / `QWEN_PE_EDIT_URL` (например, http://localhost:8100/v1);
- локально через transformers: `QWEN_PE_LOCAL=1`. Модель (~19 GB) грузится при первом
  вызове, одновременно в памяти держится только одна из двух.

Модели «думают» перед ответом (до 16–24 тыс. токенов), так что через transformers это минуты.
"""

from __future__ import annotations

import base64
import gc
import io
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "prompt_rewrite"))
import pe_core as core  # noqa: E402

CHECKPOINTS = {
    "t2i": os.environ.get("QWEN_PE_T2I_CKPT", "Qwen/Qwen-Image-2.1-PE-T2I"),
    "edit": os.environ.get("QWEN_PE_EDIT_CKPT", "Qwen/Qwen-Image-2.1-PE-I2I"),
}
SERVER_URLS = {
    "t2i": os.environ.get("QWEN_PE_T2I_URL"),
    "edit": os.environ.get("QWEN_PE_EDIT_URL"),
}
# serve.sh по умолчанию называет модель по имени папки чекпойнта.
SERVED_NAMES = {
    "t2i": os.environ.get("QWEN_PE_T2I_NAME", Path(CHECKPOINTS["t2i"]).name),
    "edit": os.environ.get("QWEN_PE_EDIT_NAME", Path(CHECKPOINTS["edit"]).name),
}
LOCAL_ENABLED = os.environ.get("QWEN_PE_LOCAL") == "1"


@dataclass
class Rewrite:
    prompt: str
    wh_ratio: str  # "16:9" и т.п. или ""
    ratio_follow: str  # "<image2>" и т.п. или ""
    thinking: str
    parse_ok: bool


class PromptEnhancer:
    def __init__(self, mock: bool) -> None:
        self.mock = mock
        self._system_prompts: dict[str, str] = {}
        self._local: tuple[str, object, object] | None = None  # (task, model, processor)
        self._lock = threading.Lock()

    def mode(self, task: str) -> str | None:
        if self.mock:
            return "mock"
        if SERVER_URLS[task]:
            return "vllm"
        if LOCAL_ENABLED:
            return "transformers"
        return None

    def status(self) -> str:
        parts = []
        for task, label in (("t2i", "T2I"), ("edit", "правки")):
            mode = self.mode(task)
            parts.append(f"{label}: {mode or 'выключено'}")
        return "Переписывание промптов — " + ", ".join(parts)

    def rewrite(self, task: str, prompt: str, images: list[Image.Image] | None = None) -> Rewrite:
        mode = self.mode(task)
        if mode is None:
            raise RuntimeError(
                "Переписывание промптов не настроено: задайте QWEN_PE_T2I_URL / QWEN_PE_EDIT_URL "
                "или QWEN_PE_LOCAL=1 (см. README)."
            )
        if mode == "mock":
            return Rewrite(f"[переписано заглушкой] {prompt}", "", "<image1>" if images else "", "", True)

        profile = core.get_profile(task)
        pe_images = [_downscale(_flatten_on_white(im), profile.image_max_pixels) for im in images or []]
        system_prompt = self._system_prompt(task)
        if mode == "vllm":
            # vLLM ждёт картинки как data: URI, transformers — как PIL.
            messages = core.build_messages(system_prompt, prompt, [_data_uri(im) for im in pe_images])
            thinking, answer = self._rewrite_vllm(task, profile, messages)
        else:
            messages = core.build_messages(system_prompt, prompt, pe_images)
            thinking, answer = self._rewrite_local(task, profile, messages)
        parsed = core.parse_answer(answer, profile)
        return Rewrite(
            prompt=parsed["positive_prompt"],
            wh_ratio=parsed["wh_ratio"] or "",
            ratio_follow=parsed["ratio_follow"] or "",
            thinking=thinking,
            parse_ok=parsed["parse_ok"],
        )

    def _system_prompt(self, task: str) -> str:
        # Системный промпт лежит внутри чекпойнта и обязан с ним совпадать — берём его оттуда.
        if task not in self._system_prompts:
            ckpt = CHECKPOINTS[task]
            if not Path(ckpt).is_dir():
                from huggingface_hub import snapshot_download

                ckpt = snapshot_download(ckpt, allow_patterns=["system_prompt.txt"])
            self._system_prompts[task] = core.load_system_prompt(None, ckpt)
        return self._system_prompts[task]

    def _rewrite_vllm(self, task, profile, messages):
        from client import _flatten, rewrite_one
        from openai import OpenAI

        client = OpenAI(base_url=SERVER_URLS[task], api_key=os.environ.get("QWEN_PE_API_KEY", "EMPTY"))
        return rewrite_one(
            client, SERVED_NAMES[task], _flatten(messages),
            temperature=profile.temperature, top_p=profile.top_p, top_k=profile.top_k,
            min_p=profile.min_p, presence_penalty=profile.presence_penalty,
            max_tokens=profile.max_new_tokens, seed=42, timeout=900.0,
        )

    def _rewrite_local(self, task, profile, messages):
        import torch
        from run_transformers import rewrite
        from transformers import AutoModelForImageTextToText, AutoProcessor

        with self._lock:
            if self._local is None or self._local[0] != task:
                self._local = None
                gc.collect()
                torch.cuda.empty_cache()
                processor = AutoProcessor.from_pretrained(CHECKPOINTS[task])
                model = AutoModelForImageTextToText.from_pretrained(
                    CHECKPOINTS[task], dtype=torch.bfloat16, low_cpu_mem_usage=True
                ).to("cuda").eval()
                self._local = (task, model, processor)
            _, model, processor = self._local
            return rewrite(
                model, processor, messages,
                max_new_tokens=profile.max_new_tokens, temperature=profile.temperature,
                top_p=profile.top_p, top_k=profile.top_k,
                presence_penalty=profile.presence_penalty, seed=42,
            )


def _flatten_on_white(image: Image.Image) -> Image.Image:
    """Как пайплайн для vision-энкодера: альфу накладываем на белый, а не отбрасываем (иначе фон чёрный)."""
    if image.mode != "RGBA":
        return image.convert("RGB")
    white = Image.new("RGB", image.size, (255, 255, 255))
    white.paste(image, mask=image.getchannel("A"))
    return white


def _data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _downscale(image: Image.Image, max_pixels: int) -> Image.Image:
    """То же, что `pe_core.load_image`, но для уже открытой картинки."""
    w, h = image.size
    if max_pixels and w * h > max_pixels:
        s = (max_pixels / float(w * h)) ** 0.5
        image = image.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return image
