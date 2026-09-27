"""Шаблоны промптов для режимов, которые у Qwen-Image-2.1 включаются формулировкой, а не флагом.

Официально задокументирован только RGBA-шаблон (README модели). Шаблоны локальных правок,
вырезания объекта и правки прозрачного слоя — наши предположения по описанию возможностей
в релизе; в интерфейсе их можно отредактировать.
"""

from __future__ import annotations

import re

from PIL import Image

# Официальный шаблон из README: "This is an RGBA image with transparency. <desc>. The image has alpha
# channel and the background is transparent."
RGBA_PREFIX = "This is an RGBA image with transparency."
RGBA_SUFFIX = "The image has alpha channel and the background is transparent."


def wrap_rgba(prompt: str) -> str:
    prompt = prompt.strip()
    if prompt.lower().startswith(RGBA_PREFIX.lower()):
        return prompt
    body = prompt.rstrip(". ")
    return f"{RGBA_PREFIX} {body}. {RGBA_SUFFIX}"


# --- Экспериментальные шаблоны ({instruction}, {subject}, {color} подставляются) ---

ANNOTATION_TEMPLATE = (
    "In the image, a region is marked with a {color} hand-drawn annotation. {instruction} "
    "Apply the change only inside the marked region, remove the annotation marks, "
    "and keep everything else unchanged."
)

MASK_TEMPLATE = (
    "<image2> is a black-and-white mask for <image1>: the white area marks the region to edit. "
    "{instruction} Apply the change only inside the white area of <image1> and keep everything "
    "outside it unchanged."
)

EXTRACT_TEMPLATE = (
    "Extract {subject} from the image onto a transparent background, keeping its exact shape, "
    "colors and details, with clean edges"
)

LAYER_EDIT_TEMPLATE = "{instruction} Keep the background transparent"

BACKGROUND_TEMPLATE = (
    "Place the object from the image into a new scene: {instruction} Keep the object's shape, colors "
    "and details exactly unchanged, and match the lighting, shadows and perspective of the scene"
)


def sentence(text: str) -> str:
    """Инструкция как законченное предложение, чтобы не сливалась с остальным шаблоном."""
    text = text.strip()
    return text if not text or text[-1] in ".!?。" else text + "."


def fill(template: str, **values: str) -> str:
    """Подставляет только известные плейсхолдеры, остальные фигурные скобки оставляет как есть."""
    return re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)


def flatten_alpha(image: Image.Image, color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Картинка с прозрачностью → RGB на однотонном фоне.

    Пайплайн отдаёт референс в VAE всеми четырьмя каналами, и модель переносит прозрачность в результат:
    фон вокруг вырезанного объекта остаётся прозрачным, что бы ни просил промпт. Под прозрачными пикселями
    у вырезок ещё и лежит случайный цвет (у RGBA-выхода модели — фиолетовый), который просто `convert("RGB")`
    вытащил бы наружу. Белый — как у vision-энкодера пайплайна, он видит RGBA-референсы поверх белого.
    """
    if image.mode not in ("RGBA", "LA", "PA") and not (image.mode == "P" and "transparency" in image.info):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = Image.new("RGB", rgba.size, color)
    canvas.paste(rgba, (0, 0), rgba)
    return canvas


NAMED_COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


def annotation_color(editor_value: dict | None) -> str:
    """Название цвета, которым рисовали в `gr.ImageEditor` (ближайший из NAMED_COLORS к среднему)."""
    total, count = [0, 0, 0], 0
    for layer in (editor_value or {}).get("layers") or []:
        if layer is None:
            continue
        rgba = layer.convert("RGBA")
        rgba.thumbnail((256, 256))
        for r, g, b, a in rgba.getdata():
            if a > 0:
                total[0] += r
                total[1] += g
                total[2] += b
                count += 1
    if not count:
        return "red"
    mean = [c / count for c in total]
    return min(NAMED_COLORS, key=lambda n: sum((m - c) ** 2 for m, c in zip(mean, NAMED_COLORS[n])))


def mask_from_layers(editor_value: dict | None) -> tuple[Image.Image, Image.Image] | None:
    """Из значения `gr.ImageEditor` достаёт (исходник RGB, маска L: белое — где рисовали)."""
    if not editor_value or editor_value.get("background") is None:
        return None
    background: Image.Image = editor_value["background"]
    mask = Image.new("L", background.size, 0)
    for layer in editor_value.get("layers") or []:
        if layer is None:
            continue
        alpha = layer.convert("RGBA").getchannel("A").point(lambda a: 255 if a > 0 else 0)
        if alpha.size != background.size:
            alpha = alpha.resize(background.size, Image.NEAREST)
        mask.paste(255, (0, 0), alpha)
    return background, mask
