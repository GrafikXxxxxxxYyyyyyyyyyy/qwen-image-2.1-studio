#!/usr/bin/env bash
# Полная установка Qwen-Image-2.1 Studio на GPU-сервер: зависимости, веса, автозапуск, публичная ссылка.
#
#   git clone https://github.com/GrafikXxxxxxxYyyyyyyyyyy/qwen-image-2.1-studio.git
#   cd qwen-image-2.1-studio && bash setup.sh
#
# Повторный запуск безопасен: уже установленное и скачанное пропускается, приложение перезапускается
# с новым кодом, а туннель (и вместе с ним публичная ссылка) остаётся прежним.
#
# На инстансе Vast.ai (есть /opt/supervisor-scripts) приложение и туннель ставятся сервисами supervisor:
# переживают падения и перезапуск контейнера, логи — в /var/log/portal/. На любой другой машине оба
# процесса запускаются в фоне через nohup, логи — в ./logs/.
#
# Переменные (необязательно):
#   QWEN_PUBLIC=0         без публичной ссылки (Cloudflare quick tunnel, без пароля — доступ у любого со ссылкой)
#   QWEN_DOWNLOAD=0       не качать веса заранее (скачаются при первом запуске приложения)
#   QWEN_TURBO=0          без turbo-LoRA
#   QWEN_PORT=17860       внутренний порт приложения (слушает только 127.0.0.1)
#   QWEN_VENV=<путь>      venv для установки; по умолчанию /venv/main (образы Vast), иначе ./.venv
# Остальные настройки приложения (QWEN_OFFLOAD, QWEN_MODEL_ID, …, см. README) на Vast читаются из
# ${WORKSPACE}/.env — после правки: supervisorctl restart qwen-studio.

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${QWEN_PORT:-17860}"
PUBLIC="${QWEN_PUBLIC:-1}"
DOWNLOAD="${QWEN_DOWNLOAD:-1}"
TURBO="${QWEN_TURBO:-1}"
MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen-Image-2.1}"
TURBO_REPO="${QWEN_TURBO_REPO:-Viggle/Qwen-Image-2.1-viggle-turbo}"
TURBO_WEIGHTS="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
VAST=0
[[ -d /opt/supervisor-scripts/utils && -d /etc/supervisor/conf.d ]] && command -v supervisorctl >/dev/null && VAST=1

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[!] %s\033[0m\n' "$*"; }
die() { printf '\033[1;31m[x] %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------- #
step "Проверка окружения"
# --------------------------------------------------------------------------- #
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/GPU: /'
else
    warn "GPU не найдена — приложение запустится в режиме заглушки (картинки ненастоящие)"
fi
cache_dir="${HF_HOME:-$HOME/.cache/huggingface}"
while [[ ! -d "$cache_dir" ]]; do cache_dir="$(dirname "$cache_dir")"; done   # кэша может ещё не быть
free_gb=$(df -BG --output=avail "$cache_dir" | tail -1 | tr -dc '0-9')
[[ "$DOWNLOAD" == 1 && "$free_gb" -lt 40 ]] && warn "Свободно ${free_gb} GB в $cache_dir, а весам нужно ~33 GB"
echo "Режим: $([[ $VAST == 1 ]] && echo 'Vast.ai (supervisor)' || echo 'обычная машина (nohup)')"

# --------------------------------------------------------------------------- #
step "Python-зависимости"
# --------------------------------------------------------------------------- #
VENV="${QWEN_VENV:-}"
if [[ -z "$VENV" ]]; then
    if [[ -x /venv/main/bin/python ]]; then VENV=/venv/main; else VENV="$APP_DIR/.venv"; fi
fi
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "Создаю venv: $VENV"
    python3 -m venv "$VENV" || die "Не удалось создать venv (нужен пакет python3-venv)"
fi
PY="$VENV/bin/python"
echo "venv: $VENV ($("$PY" --version))"
if command -v uv >/dev/null; then
    uv pip install --python "$PY" -r "$APP_DIR/requirements.txt"
else
    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -r "$APP_DIR/requirements.txt"
fi
"$PY" - <<'EOF'
import torch, diffusers, gradio
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}, доступна: {torch.cuda.is_available()}), "
      f"diffusers {diffusers.__version__}, gradio {gradio.__version__}")
from diffusers import QwenImage21Pipeline  # noqa: F401 — упадёт, если diffusers без поддержки 2.1
EOF

# --------------------------------------------------------------------------- #
if [[ "$DOWNLOAD" == 1 ]]; then
    step "Веса модели (~31 GB$([[ $TURBO == 1 ]] && echo ' + turbo-LoRA 1.3 GB'))"
    export HF_XET_HIGH_PERFORMANCE=1
    HF="$VENV/bin/hf"
    [[ -x "$HF" ]] || die "Не найден $HF (ставится вместе с huggingface_hub)"
    "$HF" download "$MODEL_ID" --quiet >/dev/null
    echo "$MODEL_ID — готово"
    if [[ "$TURBO" == 1 ]]; then
        "$HF" download "$TURBO_REPO" "$TURBO_WEIGHTS" scheduler/scheduler_config.json LICENSE --quiet >/dev/null
        echo "$TURBO_REPO — готово"
    fi
fi

# --------------------------------------------------------------------------- #
step "Автозапуск"
# --------------------------------------------------------------------------- #
CLOUDFLARED=""
if [[ "$PUBLIC" == 1 ]]; then
    CLOUDFLARED="$(command -v cloudflared || true)"
    [[ -z "$CLOUDFLARED" && -x /opt/instance-tools/bin/cloudflared ]] && CLOUDFLARED=/opt/instance-tools/bin/cloudflared
    if [[ -z "$CLOUDFLARED" ]]; then
        case "$(uname -m)" in x86_64) arch=amd64 ;; aarch64) arch=arm64 ;; *) die "cloudflared: неизвестная архитектура" ;; esac
        CLOUDFLARED="$APP_DIR/.bin/cloudflared"
        mkdir -p "$APP_DIR/.bin"
        curl -fsSL -o "$CLOUDFLARED" \
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${arch}"
        chmod +x "$CLOUDFLARED"
    fi
fi
TUNNEL_CMD="$CLOUDFLARED tunnel --no-autoupdate --url http://127.0.0.1:$PORT"

if [[ $VAST == 1 ]]; then
    APP_LOG=/var/log/portal/qwen-studio.log
    TUNNEL_LOG=/var/log/portal/qwen-studio-tunnel.log

    # Скрипты не используют exit_portal.sh: иначе сервис молча не стартует без записи в /etc/portal.yaml,
    # а для неё нужен свободный внешний порт, которого на инстансе может не быть.
    write_service() {  # имя, команда, autorestart
        cat > "/opt/supervisor-scripts/$1.sh" <<EOF
#!/bin/bash
# Сгенерировано $APP_DIR/setup.sh — правьте там и перезапустите setup.sh.
utils=/opt/supervisor-scripts/utils
. "\${utils}/logging.sh"
. "\${utils}/environment.sh"
cd "$APP_DIR"
pty $2 2>&1
EOF
        chmod +x "/opt/supervisor-scripts/$1.sh"
        cat > "/etc/supervisor/conf.d/$1.conf" <<EOF
[program:$1]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/$1.sh
autostart=true
autorestart=$3
exitcodes=0
startsecs=5
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=20
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
EOF
    }
    write_service qwen-studio "$PY app.py --host 127.0.0.1 --port $PORT" unexpected
    if [[ "$PUBLIC" == 1 ]]; then
        write_service qwen-studio-tunnel "$TUNNEL_CMD" true
    elif [[ -f /etc/supervisor/conf.d/qwen-studio-tunnel.conf ]]; then
        supervisorctl stop qwen-studio-tunnel >/dev/null 2>&1 || true
        rm -f /etc/supervisor/conf.d/qwen-studio-tunnel.conf /opt/supervisor-scripts/qwen-studio-tunnel.sh
    fi
    tunnel_was_running=$(supervisorctl status qwen-studio-tunnel 2>/dev/null | grep -c RUNNING || true)
    supervisorctl reread >/dev/null
    supervisorctl update   # запускает новые сервисы; уже работающий туннель не трогает — ссылка сохраняется
    supervisorctl restart qwen-studio >/dev/null   # подхватить новый код
    [[ "$PUBLIC" == 1 && "$tunnel_was_running" == 0 ]] && supervisorctl start qwen-studio-tunnel >/dev/null || true

    # Кнопка в портале Vast (с токеном Vast) — только если есть свободный внешний порт.
    if [[ -s /etc/portal.yaml ]] && ! grep -q "Qwen Studio" /etc/portal.yaml && command -v vast-capabilities >/dev/null; then
        ext=$(vast-capabilities 2>/dev/null | jq -r '[.instance.open_ports[]? | select(.in_use == false
              and (.self_mapped | not) and .container_port <= 65535)][0].container_port // empty' || true)
        if [[ -n "$ext" && -n "$(printenv "VAST_TCP_PORT_$ext" || true)" ]]; then
            "$PY" - "$ext" "$PORT" <<'EOF'
import sys, yaml
ext, port = int(sys.argv[1]), int(sys.argv[2])
d = yaml.safe_load(open("/etc/portal.yaml")) or {}
d.setdefault("applications", {})["Qwen Studio"] = {
    "hostname": "localhost", "external_port": ext, "internal_port": port, "open_path": "/", "name": "Qwen Studio"}
yaml.safe_dump(d, open("/etc/portal.yaml", "w"), sort_keys=False)
EOF
            supervisorctl restart caddy >/dev/null
            echo "Портал Vast: «Qwen Studio» на порту $ext"
        fi
    fi
else
    # Обычная машина: фоновые процессы. Перезапуск setup.sh заменяет приложение, туннель оставляет.
    mkdir -p "$APP_DIR/logs"
    APP_LOG="$APP_DIR/logs/app.log"
    TUNNEL_LOG="$APP_DIR/logs/tunnel.log"
    alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }
    alive "$APP_DIR/logs/app.pid" && kill "$(cat "$APP_DIR/logs/app.pid")" && sleep 3
    (cd "$APP_DIR" && nohup "$PY" app.py --host 127.0.0.1 --port "$PORT" > "$APP_LOG" 2>&1 & echo $! > "$APP_DIR/logs/app.pid")
    if [[ "$PUBLIC" == 1 ]] && ! alive "$APP_DIR/logs/tunnel.pid"; then
        nohup $TUNNEL_CMD > "$TUNNEL_LOG" 2>&1 & echo $! > "$APP_DIR/logs/tunnel.pid"
    fi
    warn "Без supervisor процессы не переживут перезагрузку машины — после неё снова запустите setup.sh"
fi

# --------------------------------------------------------------------------- #
step "Запуск (загрузка модели в видеопамять ~1–2 мин)"
# --------------------------------------------------------------------------- #
for _ in $(seq 150); do
    curl -fs -o /dev/null "http://127.0.0.1:$PORT/" && break
    if grep -q "Traceback" "$APP_LOG" 2>/dev/null; then
        tail -30 "$APP_LOG"; die "Приложение упало, лог: $APP_LOG"
    fi
    sleep 4
done
curl -fs -o /dev/null "http://127.0.0.1:$PORT/" || die "Приложение не ответило за 10 минут, лог: $APP_LOG"

LINK=""
if [[ "$PUBLIC" == 1 ]]; then
    for _ in $(seq 30); do
        LINK=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | tail -1 || true)
        [[ -n "$LINK" ]] && break
        sleep 2
    done
fi

echo
printf '\033[1;32mГотово.\033[0m\n'
[[ -n "$LINK" ]] && echo "Публичная ссылка (без пароля): $LINK"
[[ "$PUBLIC" == 1 && -z "$LINK" ]] && warn "Туннель ещё не выдал ссылку — посмотрите позже: grep trycloudflare $TUNNEL_LOG"
echo "Локально: http://127.0.0.1:$PORT  (с вашей машины: ssh -L $PORT:127.0.0.1:$PORT <сервер>)"
echo "Логи: $APP_LOG"
[[ $VAST == 1 ]] && echo "Управление: supervisorctl status | restart qwen-studio"
exit 0
