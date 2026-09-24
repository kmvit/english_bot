"""Скачивание файлов моделей. Общее для Piper и Kokoro."""
from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from . import SpeechError

log = logging.getLogger(__name__)

TIMEOUT = 600


def download(url: str, target: Path) -> None:
    """Скачать файл, если его ещё нет.

    Пишем во временный файл и переименовываем целиком: при обрыве связи на
    диске не должен остаться огрызок, который при следующем запуске сойдёт
    за готовую модель.
    """
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    log.info("Скачиваю %s", target.name)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            with tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise SpeechError(f"Не удалось скачать {target.name}: {exc}") from exc
    tmp.replace(target)
