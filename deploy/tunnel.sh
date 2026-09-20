#!/usr/bin/env bash
# SSH-туннель наружу: поднимает локальный SOCKS5, через который бот ходит
# в Telegram и OpenRouter. Параметры — из /etc/default/english-bot-tunnel.
set -euo pipefail

: "${TUNNEL_HOST:?не задан TUNNEL_HOST}"
: "${TUNNEL_USER:?не задан TUNNEL_USER}"
TUNNEL_PORT=${TUNNEL_PORT:-22}
TUNNEL_KEY=${TUNNEL_KEY:-/root/.ssh/id_rsa}
TUNNEL_BIND=${TUNNEL_BIND:-127.0.0.1:1080}

# ExitOnForwardFailure обязателен: без него ssh останется жив при занятом
# порте, systemd посчитает туннель поднятым, а бот будет стучаться в пустоту.
exec /usr/bin/ssh -N -D "$TUNNEL_BIND" \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    -i "$TUNNEL_KEY" \
    -p "$TUNNEL_PORT" \
    "$TUNNEL_USER@$TUNNEL_HOST"
