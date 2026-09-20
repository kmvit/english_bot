"""Доступ только для одного пользователя."""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject, User

log = logging.getLogger(__name__)

DENIED = "Это личный бот. Доступ закрыт."


class AllowedUserMiddleware(BaseMiddleware):
    def __init__(self, allowed_user_id: int) -> None:
        self.allowed_user_id = allowed_user_id

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user: User | None = data.get("event_from_user")
        if user is None or user.id != self.allowed_user_id:
            log.warning(
                "Отказано в доступе: user_id=%s", getattr(user, "id", None)
            )
            if isinstance(event, Message):
                await event.answer(DENIED)
            elif isinstance(event, CallbackQuery):
                await event.answer(DENIED, show_alert=True)
            return None
        return await handler(event, data)
