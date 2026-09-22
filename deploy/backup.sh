#!/usr/bin/env bash
# Резервная копия базы: весь прогресс ученика живёт в одном файле.
# Копируем через API sqlite, а не cp: бот пишет в базу в любой момент,
# и простое копирование может поймать её на середине транзакции.
set -euo pipefail

APP_DIR=${APP_DIR:-/opt/english_bot}
DB_PATH="$APP_DIR/data/bot.db"
BACKUP_DIR=${BACKUP_DIR:-$APP_DIR/backups}
KEEP=${KEEP:-14}

[ -f "$DB_PATH" ] || { echo "нет базы $DB_PATH" >&2; exit 1; }
mkdir -p "$BACKUP_DIR"

STAMP=$(date +%Y%m%d-%H%M)
TARGET="$BACKUP_DIR/bot-$STAMP.db"

python3 - "$DB_PATH" "$TARGET" <<'PY'
import sqlite3, sys

source, target = sys.argv[1], sys.argv[2]
with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
    with sqlite3.connect(target) as dst:
        src.backup(dst)
PY

gzip -f "$TARGET"
echo "копия: $TARGET.gz ($(du -h "$TARGET.gz" | cut -f1))"

# Держим последние KEEP копий, остальные удаляем.
ls -1t "$BACKUP_DIR"/bot-*.db.gz 2>/dev/null | tail -n +$((KEEP + 1)) | while read -r old; do
    rm -f "$old"
    echo "удалена старая копия: $(basename "$old")"
done
