#!/bin/bash
set -e

echo "🔒 VPN контейнер (Xray-core / VLESS+REALITY)"
python3 /vpn/build_config.py

echo "🚀 Запускаем xray..."
exec xray run -c /vpn/config.json
