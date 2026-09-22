#!/usr/bin/env bash
# Первичная установка на чистый Ubuntu. Запускать от root.
# Перед запуском положите заполненный .env в /opt/english_bot/.env
set -euo pipefail

APP_DIR=/opt/english_bot
SERVICE=english-bot

echo "==> Системные пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# ffmpeg конвертирует голосовые; libgomp1 нужен ctranslate2 и onnxruntime.
apt-get install -y -qq ffmpeg libgomp1 git python3-venv

if [ ! -d "$APP_DIR/.git" ]; then
    echo "==> Клонирую"
    git clone https://github.com/kmvit/english_bot.git "$APP_DIR"
fi
cd "$APP_DIR"

echo "==> Виртуальное окружение"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

if [ ! -f "$APP_DIR/.env" ]; then
    echo "!! Нет $APP_DIR/.env — скопируйте .env.example и заполните" >&2
    exit 1
fi

# Конфиг читает сервис под www-data, а лежит он от root. Даём доступ группе,
# но не всему миру: в файле токены бота и ключ OpenRouter.
chown root:www-data "$APP_DIR/.env"
chmod 640 "$APP_DIR/.env"

echo "==> Каталог данных"
mkdir -p "$APP_DIR/data"
chown -R www-data:www-data "$APP_DIR/data"

echo "==> Загружаю модели заранее (иначе первое голосовое будет ждать)"
sudo -u www-data "$APP_DIR/.venv/bin/python" -c "
from bot.config import load_config
from bot.speech import build_speech
import asyncio
asyncio.run(build_speech(load_config()).warmup())
print('модели готовы')
"

echo "==> Сервисы systemd"
cp "$APP_DIR/deploy/$SERVICE.service" "/etc/systemd/system/$SERVICE.service"

# Туннель ставим, только если он настроен: на машине с прямым доступом
# в интернет он не нужен и мешал бы запуску бота.
if [ -f /etc/default/english-bot-tunnel ]; then
    cp "$APP_DIR/deploy/$SERVICE-tunnel.service" "/etc/systemd/system/$SERVICE-tunnel.service"
    systemctl daemon-reload
    systemctl enable --now "$SERVICE-tunnel"
    sleep 2
    systemctl is-active --quiet "$SERVICE-tunnel" \
        && echo "==> Туннель поднят" \
        || { journalctl -u "$SERVICE-tunnel" -n 10 --no-pager; exit 1; }
fi

# Ежедневная копия базы: весь прогресс ученика — один файл.
cp "$APP_DIR/deploy/$SERVICE-backup.service" "/etc/systemd/system/"
cp "$APP_DIR/deploy/$SERVICE-backup.timer" "/etc/systemd/system/"

systemctl daemon-reload
systemctl enable --now "$SERVICE-backup.timer"
systemctl enable --now "$SERVICE"
sleep 3
systemctl is-active --quiet "$SERVICE" && echo "==> Установлен и работает" || {
    journalctl -u "$SERVICE" -n 20 --no-pager
    exit 1
}
