#!/usr/bin/env bash
# Обновление бота на сервере: код, зависимости, перезапуск.
# Запускать на сервере от root: /opt/english_bot/deploy/deploy.sh
set -euo pipefail

APP_DIR=/opt/english_bot
SERVICE=english-bot

cd "$APP_DIR"

echo "==> Забираю код"
git pull --ff-only

echo "==> Доустанавливаю зависимости"
.venv/bin/pip install -q -r requirements.txt

# Модели и база принадлежат сервису, а git-операции идут от root.
echo "==> Права на данные"
mkdir -p "$APP_DIR/data"
chown -R www-data:www-data "$APP_DIR/data"

echo "==> Перезапускаю сервис"
systemctl restart "$SERVICE"
sleep 3
systemctl is-active --quiet "$SERVICE" && echo "==> Работает" || {
    echo "==> НЕ ЗАПУСТИЛСЯ, последние строки журнала:"
    journalctl -u "$SERVICE" -n 20 --no-pager
    exit 1
}
