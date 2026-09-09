FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates unzip python3 \
    && rm -rf /var/lib/apt/lists/*

# Официальный Xray-core (нужен для VLESS Reality + Vision + пост-квантовой
# верификации mldsa65 — mihomo/clash-meta пока это не поддерживают).
# Берём "latest" через редирект GitHub, чтобы не зависеть от устаревшего тега.
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "aarch64" ]; then FILE="Xray-linux-arm64-v8a.zip"; \
    else FILE="Xray-linux-64.zip"; fi && \
    curl -fsSL "https://github.com/XTLS/Xray-core/releases/latest/download/${FILE}" -o /tmp/xray.zip && \
    unzip -o /tmp/xray.zip -d /usr/local/bin xray geoip.dat geosite.dat && \
    chmod +x /usr/local/bin/xray && \
    rm /tmp/xray.zip

WORKDIR /vpn
COPY entrypoint.sh /entrypoint.sh
COPY build_config.py /vpn/build_config.py
RUN chmod +x /entrypoint.sh

# 1080 = SOCKS5, 1081 = HTTP — доступны другим контейнерам по имени "vpn"
EXPOSE 1080 1081

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=10s \
    CMD curl -sf -x socks5h://127.0.0.1:1080 https://www.gstatic.com/generate_204 -o /dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
