"""Gradio-интерфейс для Qwen-Image-2.1: генерация, редактирование, референсы, локальные правки, RGBA.

Запуск:
    python app.py                      # на GPU-сервере: настоящая модель
    QWEN_MOCK=1 python app.py          # где угодно: заглушка для отладки интерфейса
    python app.py --share --auth user:pass
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image

import prompts
from backend import (
    ASPECT_RATIOS,
    MAX_REFERENCE_IMAGES,
    GenRequest,
    load_backend,
    size_for_ratio,
    size_like_image,
)
from enhancer import PromptEnhancer

OUTPUT_DIR = Path(os.environ.get("QWEN_OUTPUT_DIR", "outputs"))
RESOLUTIONS = [1024, 1536, 2048]
MAX_SEED = 2**31 - 1
SIZE_FOLLOW = "Как у изображения №"
SIZE_RATIO = "Соотношение сторон"

backend = load_backend()
enhancer = PromptEnhancer(mock=backend.is_mock)


# --------------------------------------------------------------------------- #
# Общая часть: запуск генерации
# --------------------------------------------------------------------------- #
def run(prompt: str, images: list[Image.Image], size: tuple[int, int] | None, ref_resolution: int,
        settings: list, progress: gr.Progress):
    steps, seed, randomize, num_images, cfg_scale, negative, kv_cache = settings
    if not prompt.strip():
        raise gr.Error("Пустой промпт")
    seed = random.randint(0, MAX_SEED) if randomize else int(seed)
    req = GenRequest(
        prompt=prompt.strip(),
        images=images,
        width=size[0] if size else None,
        height=size[1] if size else None,
        steps=int(steps),
        seed=seed,
        negative_prompt=negative or "",
        true_cfg_scale=float(cfg_scale),
        output_resolution=int(ref_resolution),
        use_kv_cache=bool(kv_cache),
        num_images=int(num_images),
    )
    width, height = req.resolved_size()
    progress(0, desc=f"Генерация {width}×{height}")
    started = time.time()
    try:
        results = backend.generate(req, progress=lambda i, n: progress((i, n), desc="Денойзинг"))
    except Exception as e:  # OOM и прочее — показать в интерфейсе, а не уронить очередь
        raise gr.Error(f"{type(e).__name__}: {e}") from e
    elapsed = time.time() - started

    saved = save_outputs(results, req)
    modes = sorted({im.mode for im in results})
    info = (
        f"**{width}×{height}** · seed `{seed}` · {req.steps} шагов · {elapsed:.1f} с · {'/'.join(modes)}"
        + (f" · референсов: {len(images)}" if images else "")
        + (f" · CFG {req.true_cfg_scale}" if req.negative_prompt and req.true_cfg_scale > 1 else "")
        + (f"  \nСохранено: `{saved}`" if saved else "")
        + ("  \n⚠️ Заглушка: картинки ненастоящие" if backend.is_mock else "")
    )
    return results, req.prompt, info, seed


def save_outputs(results: list[Image.Image], req: GenRequest) -> str | None:
    if backend.is_mock:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{req.seed}"
    for i, im in enumerate(results):
        im.save(OUTPUT_DIR / f"{stem}_{i}.png")
    meta = {k: v for k, v in vars(req).items() if k != "images"}
    meta["num_reference_images"] = len(req.images)
    meta["size"] = req.resolved_size()
    (OUTPUT_DIR / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return str(OUTPUT_DIR / stem)


def gallery_images(value) -> list[Image.Image]:
    """Значение интерактивной галереи → список PIL в порядке загрузки."""
    images = []
    for item in value or []:
        item = item[0] if isinstance(item, (tuple, list)) else item
        images.append(item if isinstance(item, Image.Image) else Image.open(item))
    return images


def output_size(images, mode, follow_idx, ratio, resolution) -> tuple[int, int]:
    if mode == SIZE_FOLLOW:
        idx = int(follow_idx)
        if not 1 <= idx <= len(images):
            raise gr.Error(f"Нет изображения №{idx}: загружено {len(images)}")
        return size_like_image(images[idx - 1], int(resolution))
    return size_for_ratio(ratio, int(resolution))


# --------------------------------------------------------------------------- #
# Обработчики вкладок
# --------------------------------------------------------------------------- #
def t2i(prompt, rgba, ratio, resolution, *settings, progress=gr.Progress()):
    prompt = prompts.wrap_rgba(prompt) if rgba else prompt
    return run(prompt, [], size_for_ratio(ratio, int(resolution)), 1024, list(settings), progress)


def rewrite_t2i(prompt):
    if not prompt.strip():
        raise gr.Error("Пустой промпт")
    r = enhancer.rewrite("t2i", prompt)
    ratio = gr.update(value=r.wh_ratio) if r.wh_ratio in ASPECT_RATIOS else gr.update()
    return r.prompt, ratio, format_thinking(r)


def edit(gallery, instruction, rgba, size_mode, follow_idx, ratio, out_res, ref_res, *settings,
         progress=gr.Progress()):
    images = gallery_images(gallery)
    if not images:
        raise gr.Error("Загрузите хотя бы одно изображение")
    if len(images) > MAX_REFERENCE_IMAGES:
        raise gr.Error(f"Модель принимает не больше {MAX_REFERENCE_IMAGES} изображений, загружено {len(images)}")
    prompt = prompts.wrap_rgba(instruction) if rgba else instruction
    size = output_size(images, size_mode, follow_idx, ratio, out_res)
    return run(prompt, images, size, ref_res, list(settings), progress)


def rewrite_edit(gallery, instruction):
    images = gallery_images(gallery)
    if not images or not instruction.strip():
        raise gr.Error("Нужны изображения и инструкция")
    r = enhancer.rewrite("edit", instruction, images)
    mode, idx, ratio = gr.update(), gr.update(), gr.update()
    if follow := re.fullmatch(r"<image(\d+)>", r.ratio_follow):
        mode, idx = SIZE_FOLLOW, int(follow.group(1))
    elif r.wh_ratio in ASPECT_RATIOS:
        mode, ratio = SIZE_RATIO, r.wh_ratio
    return r.prompt, mode, idx, ratio, format_thinking(r)


def format_thinking(r) -> str:
    head = "" if r.parse_ok else "⚠️ Ответ PE не распарсился как JSON — в промпт попал сырой текст.\n\n"
    extra = f"wh_ratio: {r.wh_ratio or '—'} · ratio_follow: {r.ratio_follow or '—'}\n\n"
    return head + extra + (r.thinking or "(рассуждение пустое)")


def local_edit(editor, mode, instruction, template, out_res, *settings, progress=gr.Progress()):
    extracted = prompts.mask_from_layers(editor)
    if extracted is None:
        raise gr.Error("Загрузите изображение и отметьте область кистью")
    background, mask = extracted
    if mask.getextrema()[1] == 0:
        raise gr.Error("На изображении ничего не нарисовано — отметьте область кистью")
    if not instruction.strip():
        raise gr.Error("Опишите, что сделать с отмеченной областью")
    background = background.convert("RGB")
    size = size_like_image(background, int(out_res))

    if mode == "mask":
        prompt = prompts.fill(template, instruction=prompts.sentence(instruction))
        images, preview = [background, mask.convert("RGB")], mask
    else:
        composite = editor["composite"].convert("RGB")
        color = prompts.annotation_color(editor)
        prompt = prompts.fill(template, instruction=prompts.sentence(instruction), color=color)
        images, preview = [composite], composite
    results, used_prompt, info, seed = run(prompt, images, size, out_res, list(settings), progress)
    return results, used_prompt, info, seed, preview


def transparency(image, mode, text, template, out_res, *settings, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Загрузите изображение")
    if mode == "extract":
        body = prompts.fill(template, subject=text.strip() or "the main subject")
    else:
        if not text.strip():
            raise gr.Error("Опишите правку слоя")
        body = prompts.fill(template, instruction=prompts.sentence(text))
    size = size_like_image(image, int(out_res))
    return run(prompts.wrap_rgba(body), [image], size, out_res, list(settings), progress)


# --------------------------------------------------------------------------- #
# Интерфейс
# --------------------------------------------------------------------------- #
CSS = """
.checker img { background: repeating-conic-gradient(#d9d9d9 0 25%, #ffffff 0 50%) 50% / 20px 20px; }
.mock-banner { background: rgba(255, 193, 7, 0.15); border: 1px solid rgba(255, 193, 7, 0.6);
               padding: 8px 12px; border-radius: 8px; }
"""

HELP_MD = f"""
### Советы

**Несколько референсов.** Модель читает картинки по порядку, в промпте ссылайтесь на них тегами
`<image1>`, `<image2>`, … (так пишет и официальный переписчик промптов). Пример:
`Put the cat from <image1> on the sofa in <image2>, keep the cat's appearance unchanged.`
До {MAX_REFERENCE_IMAGES} изображений.

**Прозрачность.** Официальный шаблон: `{prompts.RGBA_PREFIX} <описание>. {prompts.RGBA_SUFFIX}`.
Галочка «Прозрачный фон» оборачивает промпт в него сама.

**Разрешение референсов.** Каждый референс приводится к площади N×N и превращается в токены
(1024² → 4096 токенов, 2048² → 16384). С большим количеством референсов держите 1024.

**Без CFG.** Модель рассчитана на генерацию без guidance. Negative prompt работает только при
CFG > 1 и удваивает время шага.

**KV-кэш.** Промпт и референсы считаются один раз и переиспользуются на всех шагах.
Включённый и выключенный кэш дают разные (одинаково корректные) картинки при одном seed.

**Экспериментальные режимы.** Шаблоны «Локальной правки» и «Прозрачности» (кроме генерации RGBA
по тексту) — наши предположения: Qwen показывает эти возможности, но не публикует формат промпта.
Шаблоны можно править прямо в интерфейсе.
"""


def settings_sidebar():
    with gr.Sidebar(open=True):
        gr.Markdown("### Параметры генерации")
        steps = gr.Slider(4, 80, value=40, step=1, label="Шаги", info="Рекомендовано: 40")
        seed = gr.Number(value=42, precision=0, minimum=0, maximum=MAX_SEED, label="Seed")
        randomize = gr.Checkbox(value=True, label="Случайный seed при каждом запуске")
        num_images = gr.Slider(1, 4, value=1, step=1, label="Картинок за раз")
        kv_cache = gr.Checkbox(value=True, label="KV-кэш префикса",
                               info="Ускоряет шаги; влияет на точное воспроизведение seed")
        with gr.Accordion("CFG (по умолчанию выключен)", open=False):
            cfg_scale = gr.Slider(1.0, 8.0, value=1.0, step=0.1, label="true_cfg_scale",
                                  info="> 1 включает CFG, нужен negative prompt")
            negative = gr.Textbox(label="Negative prompt", lines=2)
        gr.Markdown(f"<small>{backend.description}<br>{enhancer.status()}</small>")
    return [steps, seed, randomize, num_images, cfg_scale, negative, kv_cache], seed


def result_column():
    gallery = gr.Gallery(label="Результат", format="png", type="pil", columns=2, height=560,
                         elem_classes="checker", interactive=False)
    info = gr.Markdown()
    used_prompt = gr.Textbox(label="Промпт, отправленный в модель", lines=3, interactive=False)
    return gallery, used_prompt, info


def pe_button(task: str):
    enabled = enhancer.mode(task) is not None
    return gr.Button("✨ Переписать промпт (Qwen PE)" if enabled else "Переписчик промптов не настроен",
                     interactive=enabled)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Qwen-Image-2.1 Studio") as demo:
        gr.Markdown("# Qwen-Image-2.1 Studio")
        if backend.is_mock:
            gr.Markdown("⚠️ **Режим заглушки** — модель не загружена, результаты — плейсхолдеры "
                        "для проверки интерфейса.", elem_classes="mock-banner")
        settings, seed_box = settings_sidebar()

        # ---------------- Текст → изображение ----------------
        with gr.Tab("Текст → изображение"):
            with gr.Row():
                with gr.Column():
                    t_prompt = gr.Textbox(label="Промпт", lines=5, placeholder=(
                        'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement'))
                    t_rgba = gr.Checkbox(label="Прозрачный фон (RGBA)")
                    t_ratio = gr.Radio(list(ASPECT_RATIOS), value="1:1", label="Соотношение сторон")
                    t_res = gr.Radio(RESOLUTIONS, value=2048, label="Разрешение (длина стороны квадрата)",
                                     info="Модель родная для 2K; 1024 — быстрее и легче")
                    with gr.Row():
                        t_rewrite = pe_button("t2i")
                        t_go = gr.Button("Сгенерировать", variant="primary")
                    with gr.Accordion("Рассуждение переписчика промптов", open=False):
                        t_thinking = gr.Textbox(lines=10, show_label=False, interactive=False)
                with gr.Column():
                    t_out, t_used, t_info = result_column()
            t_rewrite.click(rewrite_t2i, [t_prompt], [t_prompt, t_ratio, t_thinking])
            t_go.click(t2i, [t_prompt, t_rgba, t_ratio, t_res, *settings], [t_out, t_used, t_info, seed_box])

        # ---------------- Редактирование / референсы ----------------
        with gr.Tab("Редактирование и референсы"):
            with gr.Row():
                with gr.Column():
                    e_images = gr.Gallery(label=f"Изображения (1–{MAX_REFERENCE_IMAGES}, порядок = <image1>, <image2>, …)",
                                          type="pil", format="png", interactive=True, columns=5, height=260)
                    e_prompt = gr.Textbox(label="Инструкция", lines=4, placeholder=(
                        "Change the background to a sunset beach\n"
                        "или: The people from <image1> and <image2> sit together at a cafe table"))
                    e_rgba = gr.Checkbox(label="Результат с прозрачным фоном (RGBA)")
                    with gr.Row():
                        e_size_mode = gr.Radio([SIZE_FOLLOW, SIZE_RATIO], value=SIZE_FOLLOW, label="Размер вывода")
                        e_follow = gr.Number(value=1, precision=0, minimum=1, maximum=MAX_REFERENCE_IMAGES,
                                             label="№ изображения")
                    e_ratio = gr.Radio(list(ASPECT_RATIOS), value="1:1", label="Соотношение сторон")
                    with gr.Row():
                        e_out_res = gr.Radio(RESOLUTIONS, value=1024, label="Разрешение вывода")
                        e_ref_res = gr.Radio(RESOLUTIONS, value=1024, label="Разрешение референсов")
                    with gr.Row():
                        e_rewrite = pe_button("edit")
                        e_go = gr.Button("Сгенерировать", variant="primary")
                    with gr.Accordion("Рассуждение переписчика промптов", open=False):
                        e_thinking = gr.Textbox(lines=10, show_label=False, interactive=False)
                with gr.Column():
                    e_out, e_used, e_info = result_column()
            e_rewrite.click(rewrite_edit, [e_images, e_prompt], [e_prompt, e_size_mode, e_follow, e_ratio, e_thinking])
            e_go.click(edit, [e_images, e_prompt, e_rgba, e_size_mode, e_follow, e_ratio, e_out_res, e_ref_res,
                              *settings], [e_out, e_used, e_info, seed_box])

        # ---------------- Локальная правка ----------------
        with gr.Tab("Локальная правка"):
            gr.Markdown("Обведите или закрасьте область кистью, опишите правку. "
                        "*Экспериментально: формат промпта подобран нами.*")
            with gr.Row():
                with gr.Column():
                    l_editor = gr.ImageEditor(
                        label="Изображение и пометка", type="pil", image_mode="RGBA", format="png", height=520,
                        brush=gr.Brush(colors=["#FF0000", "#00FF00", "#0000FF", "#FFFF00"],
                                       default_color="#FF0000", default_size=12),
                    )
                    l_mode = gr.Radio([("Пометка на картинке (круг, штрихи)", "annotation"),
                                       ("Отдельная маска (закрашенное → белое)", "mask")],
                                      value="annotation", label="Как передать область")
                    l_prompt = gr.Textbox(label="Что сделать", lines=3,
                                          placeholder="Replace the cup with a glass of orange juice.")
                    l_template = gr.Textbox(label="Шаблон промпта", value=prompts.ANNOTATION_TEMPLATE, lines=4)
                    l_out_res = gr.Radio(RESOLUTIONS, value=1024, label="Разрешение")
                    l_go = gr.Button("Применить", variant="primary")
                with gr.Column():
                    l_out, l_used, l_info = result_column()
                    l_preview = gr.Image(label="Что ушло в модель как пометка/маска", type="pil",
                                         interactive=False, height=240)
            l_mode.change(lambda m: prompts.MASK_TEMPLATE if m == "mask" else prompts.ANNOTATION_TEMPLATE,
                          l_mode, l_template)
            l_go.click(local_edit, [l_editor, l_mode, l_prompt, l_template, l_out_res, *settings],
                       [l_out, l_used, l_info, seed_box, l_preview])

        # ---------------- Прозрачность ----------------
        with gr.Tab("Прозрачность"):
            gr.Markdown("Генерация RGBA по тексту — галочка на первой вкладке. Здесь — вырезание объекта "
                        "из фото и правка прозрачного слоя. *Шаблоны экспериментальные.*")
            with gr.Row():
                with gr.Column():
                    a_image = gr.Image(label="Изображение (PNG с альфой для правки слоя)", type="pil",
                                       image_mode="RGBA", format="png", height=400, elem_classes="checker")
                    a_mode = gr.Radio([("Вырезать объект из фото", "extract"),
                                       ("Изменить прозрачный слой", "layer")], value="extract", label="Режим")
                    a_text = gr.Textbox(label="Что вырезать", placeholder="the red sports car")
                    a_template = gr.Textbox(label="Шаблон (обернётся в RGBA-шаблон)",
                                            value=prompts.EXTRACT_TEMPLATE, lines=3)
                    a_out_res = gr.Radio(RESOLUTIONS, value=1024, label="Разрешение")
                    a_go = gr.Button("Сгенерировать", variant="primary")
                with gr.Column():
                    a_out, a_used, a_info = result_column()

            def switch_alpha_mode(mode):
                if mode == "extract":
                    return (gr.update(label="Что вырезать", placeholder="the red sports car", value=""),
                            prompts.EXTRACT_TEMPLATE)
                return (gr.update(label="Правка слоя", placeholder='Change the text on the sticker to "HELLO"',
                                  value=""), prompts.LAYER_EDIT_TEMPLATE)

            a_mode.change(switch_alpha_mode, a_mode, [a_text, a_template])
            a_go.click(transparency, [a_image, a_mode, a_text, a_template, a_out_res, *settings],
                       [a_out, a_used, a_info, seed_box])

        with gr.Tab("Справка"):
            gr.Markdown(HELP_MD)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    ap.add_argument("--share", action="store_true", help="Публичная ссылка *.gradio.live")
    ap.add_argument("--auth", help="Логин и пароль в виде user:pass (обязательно с --share)")
    args = ap.parse_args()
    auth = tuple(args.auth.split(":", 1)) if args.auth else None
    if args.share and not auth:
        print("[app] ВНИМАНИЕ: публичная ссылка без --auth — GPU доступен любому, у кого есть ссылка")

    demo = build_ui()
    # Один GPU — одна генерация за раз, остальные ждут в очереди.
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, auth=auth,
                theme=gr.themes.Soft(), css=CSS)


if __name__ == "__main__":
    main()
