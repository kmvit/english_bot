from __future__ import annotations

import pytest
import pytest_asyncio

from bot.config import Config, Prices
from bot.db import Database


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        telegram_token="test-token",
        openrouter_api_key="test-key",
        azure_speech_key="",
        azure_speech_region="",
        allowed_user_id=1,
        llm_model="openai/test-model",
        llm_fallback_model="openai/test-mini",
        db_path=tmp_path / "test.db",
        history_limit=6,
        level_check_every=3,
        prices=Prices(),
    )


@pytest_asyncio.fixture
async def db(config: Config):
    database = Database(config.db_path)
    await database.connect()
    yield database
    await database.close()
