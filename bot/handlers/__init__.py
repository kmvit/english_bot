"""Роутеры бота."""
from aiogram import Router

from . import chat, common, drill, stats


def build_router() -> Router:
    """Собрать корневой роутер. Порядок важен: команды раньше свободного текста."""
    router = Router(name="root")
    router.include_router(common.build())
    router.include_router(drill.build())
    router.include_router(stats.build())
    router.include_router(chat.build())
    return router


__all__ = ["build_router"]
