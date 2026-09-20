"""Тесты сборки сессии Telegram: прокси и принудительный IPv4."""
from __future__ import annotations

import socket
from dataclasses import replace

from aiohttp import TCPConnector

from bot.__main__ import build_session
from bot.config import Config


def test_no_session_when_nothing_to_configure(config: Config):
    """Без прокси и без IPv4 своя сессия не нужна — aiogram соберёт свою."""
    assert build_session(config) is None


def test_proxy_switches_connector_to_socks(config: Config):
    session = build_session(replace(config, proxy_url="socks5://127.0.0.1:1080"))

    assert session is not None
    # aiohttp-socks подменяет коннектор — обычный TCPConnector через SOCKS не ходит.
    assert session._connector_type is not TCPConnector
    assert "Proxy" in session._connector_type.__name__


def test_force_ipv4_without_proxy(config: Config):
    session = build_session(replace(config, telegram_force_ipv4=True))

    assert session is not None
    assert session._connector_init["family"] == socket.AF_INET


def test_proxy_and_ipv4_together(config: Config):
    session = build_session(
        replace(config, proxy_url="socks5://127.0.0.1:1080", telegram_force_ipv4=True)
    )

    assert session is not None
    assert "Proxy" in session._connector_type.__name__
    assert session._connector_init["family"] == socket.AF_INET
