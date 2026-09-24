# Qwen-Image-2.1 Studio

Gradio-интерфейс ко всем возможностям [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1):

| Вкладка | Что делает | Статус |
|---|---|---|
| Текст → изображение | Любой размер от 256 до 3072 px по каждой стороне (ползунки, кратно 32), RGBA по галочке | официальный API |
| Редактирование и референсы | Правка по инструкции, композиция из 1–10 картинок (`<image1>`, `<image2>`…), размер «как у №N» или вручную | официальный API |
| Локальная правка | Обвести/закрасить область кистью → пометкой на картинке или отдельной маской | **шаблон промпта — наша догадка** |
| Прозрачность | Вырезать объект из фото, изменить прозрачный слой | **шаблон промпта — наша догадка** |
| ✨ Переписать промпт | Официальные модели Qwen-Image-2.1-PE-T2I / PE-I2I, заодно подбирают пропорции (пересчитываются в ширину и высоту) | нужен отдельный запуск (см. ниже) |

Общие параметры (⚡ turbo, шаги, seed, число картинок, KV-кэш, CFG + negative prompt) — в боковой панели.
Каждая генерация на сервере сохраняется в `outputs/` вместе с JSON параметров.

## Файлы

```
app.py            интерфейс и обработчики вкладок
backend.py        DiffusersBackend (CUDA) и MockBackend (заглушка), расчёт размеров
prompts.py        RGBA-шаблон и экспериментальные шаблоны, маска из слоёв редактора
enhancer.py       переписывание промптов: vLLM-сервер или transformers
prompt_rewrite/   официальный код Qwen без изменений (см. NOTICE.md)
```

## Отладка интерфейса без GPU (Mac)

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -r requirements-mock.txt
QWEN_MOCK=1 .venv/bin/python app.py
```

Вместо модели рисуются плейсхолдеры нужного размера с параметрами запроса — видно, какой промпт,
размер и референсы ушли бы в модель.

## Быстрая установка на GPU-сервер

```bash
git clone https://github.com/GrafikXxxxxxxYyyyyyyyyyy/qwen-image-2.1-studio.git
cd qwen-image-2.1-studio && bash setup.sh
```

`setup.sh` ставит зависимости, качает веса (~33 GB вместе с turbo-LoRA), запускает приложение и открывает
публичную ссылку через Cloudflare quick tunnel (**без пароля**: GPU доступен любому, у кого есть ссылка;
`QWEN_PUBLIC=0` — без ссылки). В конце печатает адрес. На Vast.ai приложение и туннель ставятся сервисами
supervisor (`qwen-studio`, `qwen-studio-tunnel`), на других машинах — фоновыми процессами с логами в `logs/`.
Повторный запуск безопасен: обновляет код и перезапускает приложение, ссылка остаётся прежней.
Опции — в шапке `setup.sh`.

## Запуск на GPU-сервере вручную

**Железо:**
- диск: ~31 GB под модель + 1.3 GB turbo-LoRA (+ ~19 GB на каждую PE-модель);
- 80 GB VRAM (A100/H100) — всё в видеопамяти (`none`), влезает и локальный PE (оценка, не проверял);
- 48 GB (A6000/L40S) — `encoder`: трансформер и VAE в видеопамяти, текстовый энкодер в RAM (выбирается сам);
- 24 GB (3090/4090) — `model` или `sequential`, заметно медленнее; нужно ≥ 64 GB RAM (оценка, не проверял).

Замеры на RTX A6000 48 GB (`encoder`, одна картинка, включая энкодер и декодирование):

| | 1024×1024 | 2752×1536 |
|---|---|---|
| база, 40 шагов | 32 с | ~160 с |
| ⚡ turbo, 6 шагов | 7–9 с | 30 с |
| (для сравнения: база с `model`) | 70 с | 197 с |

```bash
# на сервере
rsync -av --exclude .venv --exclude outputs ./ user@server:~/qwen-studio/   # с вашей машины
cd ~/qwen-studio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf download Qwen/Qwen-Image-2.1        # необязательно: скачать веса заранее (~35 GB)
python app.py --host 127.0.0.1 --port 7860
```

Открыть с вашей машины — через SSH-туннель (безопаснее публичной ссылки):

```bash
ssh -L 7860:localhost:7860 user@server
```

и дальше http://localhost:7860. Альтернатива — `python app.py --share --auth логин:пароль`
(публичная ссылка `*.gradio.live`; без `--auth` GPU доступен любому, у кого есть ссылка).

### Переменные окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `QWEN_MOCK` | — | `1` — заглушка вместо модели |
| `QWEN_MODEL_ID` | `Qwen/Qwen-Image-2.1` | id на Hub или локальная папка |
| `QWEN_OFFLOAD` | `auto` | `none` / `encoder` / `model` / `sequential`; `auto` = `none` при ≥ 64 GB VRAM, `encoder` при ≥ 40 GB, иначе `model` |
| `QWEN_TURBO` | `1` | `0` — не загружать turbo-LoRA ([Viggle/Qwen-Image-2.1-viggle-turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)) |
| `QWEN_COMPILE` | — | `1` — flex attention + `torch.compile`. На torch 2.11 падает в Inductor (`'function' object has no attribute 'graph'`) |
| `QWEN_OUTPUT_DIR` | `outputs` | куда сохранять результаты |
| `QWEN_PE_T2I_URL`, `QWEN_PE_EDIT_URL` | — | адреса vLLM-серверов переписчика, например `http://localhost:8100/v1` |
| `QWEN_PE_T2I_NAME`, `QWEN_PE_EDIT_NAME` | имя папки чекпойнта | `served-model-name` на vLLM-сервере |
| `QWEN_PE_LOCAL` | — | `1` — переписчик через transformers в том же процессе |

## Переписчик промптов (необязательно)

Это отдельные VLM на ~9B параметров, которые перед ответом «думают» (тысячи токенов), поэтому
через transformers ответ занимает минуты. Рекомендуемый путь — vLLM, **в отдельном окружении**
(официальный `requirements.txt` переписчика фиксирует `transformers==5.4`, а основной модели
нужна ≥ 5.17) и лучше на отдельной видеокарте:

```bash
git clone https://github.com/QwenLM/Qwen-Image-2.1 && cd Qwen-Image-2.1/prompt_rewrite
python3 -m venv .venv-pe && source .venv-pe/bin/activate && pip install -r requirements.txt
hf download Qwen/Qwen-Image-2.1-PE-T2I --local-dir ~/models/Qwen-Image-2.1-PE-T2I
hf download Qwen/Qwen-Image-2.1-PE-I2I --local-dir ~/models/Qwen-Image-2.1-PE-I2I
CKPT=~/models/Qwen-Image-2.1-PE-T2I PORT=8100 GPUS=1 bash serve.sh &
CKPT=~/models/Qwen-Image-2.1-PE-I2I PORT=8101 GPUS=2 bash serve.sh &
```

Затем запустить студию с `QWEN_PE_T2I_URL=http://localhost:8100/v1 QWEN_PE_EDIT_URL=http://localhost:8101/v1`.

Если GPU один и на 80 GB — можно `QWEN_PE_LOCAL=1` (модель грузится при первом нажатии,
в памяти держится одна из двух).

## Про экспериментальные режимы

Qwen показывает локальные правки (круги, пометки, маски) и вырезание объекта, но формат промпта
для них не публикует. Шаблоны в `prompts.py` — отправная точка; их можно править прямо в интерфейсе.
Для маски модель получает два изображения: `<image1>` — исходник, `<image2>` — чёрно-белая маска.
Что именно ушло в модель, видно в поле «Промпт, отправленный в модель» и в превью пометки/маски.

## Лицензия

Модель и код из `prompt_rewrite/` — Qwen Research License: **только некоммерческое использование**.
