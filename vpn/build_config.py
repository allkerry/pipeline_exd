#!/usr/bin/env python3
"""
Строит config.json для Xray-core из ссылки vless://... (REALITY).
Читает ссылку из переменной окружения VLESS_URL.

Поддерживает все актуальные поля REALITY-ссылок, включая:
  - flow (xtls-rprx-vision)
  - fp (uTLS fingerprint)
  - pbk (публичный ключ X25519)
  - sid (shortId)
  - sni (serverName)
  - spx (spiderX)
  - pqv (пост-квантовый верификационный ключ ML-DSA-65, поле mldsa65Verify)
"""
import json
import os
import sys
from urllib.parse import urlparse, parse_qs, unquote

VLESS_URL = os.environ.get("VLESS_URL", "").strip()
SOCKS_PORT = int(os.environ.get("SOCKS_PORT", "1080"))
HTTP_PORT = int(os.environ.get("HTTP_PORT", "1081"))
LOG_LEVEL = os.environ.get("XRAY_LOG_LEVEL", "warning")
# Пост-квантовая верификация REALITY (ML-DSA-65 / поле "pqv" в ссылке) —
# новая, иногда капризная фича Xray-core (известны случаи, когда она мешает
# установить соединение). Если VPN не поднимается — выстави
# DISABLE_PQV=true в .env и пересобери контейнер.
DISABLE_PQV = os.environ.get("DISABLE_PQV", "false").strip().lower() == "true"

if not VLESS_URL:
    print("❌ VLESS_URL не задан (проверь .env)", file=sys.stderr)
    sys.exit(1)

if not VLESS_URL.startswith("vless://"):
    print("❌ Ожидается ссылка вида vless://...", file=sys.stderr)
    sys.exit(1)

p = urlparse(VLESS_URL)
q = parse_qs(p.query)


def qget(key, default=""):
    return q.get(key, [default])[0]


uuid = p.username
server = p.hostname
port = p.port
flow = qget("flow")
fp = qget("fp", "chrome")
pbk = qget("pbk")
sid = qget("sid", "")
sni = qget("sni")
spx = unquote(qget("spx", "/"))
security = qget("security", "reality")
network = qget("type", "tcp")
encryption = qget("encryption", "none")
pqv = qget("pqv", "")
remark = unquote(p.fragment) if p.fragment else "vpn"

if not (uuid and server and port and pbk and sni):
    print("❌ В ссылке не хватает обязательных полей (uuid/server/port/pbk/sni)", file=sys.stderr)
    sys.exit(1)

if security != "reality":
    print(f"❌ Поддерживается только security=reality, получено: {security!r}", file=sys.stderr)
    sys.exit(1)

reality_settings = {
    "show": False,
    "fingerprint": fp,
    "serverName": sni,
    "publicKey": pbk,
    "shortId": sid,
    "spiderX": spx,
}
if pqv and not DISABLE_PQV:
    # Пост-квантовая верификация REALITY (ML-DSA-65).
    reality_settings["mldsa65Verify"] = pqv
elif pqv and DISABLE_PQV:
    print("⚠️  pqv найден в ссылке, но DISABLE_PQV=true — верификация ML-DSA-65 отключена")

vless_user = {
    "id": uuid,
    "encryption": encryption or "none",
}
if flow:
    vless_user["flow"] = flow

config = {
    "log": {"loglevel": LOG_LEVEL},
    "inbounds": [
        {
            "listen": "0.0.0.0",
            "port": SOCKS_PORT,
            "protocol": "socks",
            "settings": {"auth": "noauth", "udp": True},
            "tag": "socks-in",
        },
        {
            "listen": "0.0.0.0",
            "port": HTTP_PORT,
            "protocol": "http",
            "settings": {},
            "tag": "http-in",
        },
    ],
    "outbounds": [
        {
            "protocol": "vless",
            "tag": "proxy",
            "settings": {
                "vnext": [
                    {
                        "address": server,
                        "port": port,
                        "users": [vless_user],
                    }
                ]
            },
            "streamSettings": {
                "network": network,
                "security": "reality",
                "realitySettings": reality_settings,
            },
        },
        {"protocol": "freedom", "tag": "direct"},
        {"protocol": "blackhole", "tag": "block"},
    ],
    "routing": {
        "domainStrategy": "IPIfNonMatch",
        "rules": [
            {"type": "field", "inboundTag": ["socks-in", "http-in"], "outboundTag": "proxy"}
        ],
    },
}

with open("/vpn/config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"✅ Конфиг Xray собран: {remark} → {server}:{port} (sni={sni}, fp={fp}, pq={'да' if pqv else 'нет'})")
