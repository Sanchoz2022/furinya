#!/bin/bash
# Установка клиента Mihomo (Clash.Meta) в LXC — прокси для внешних API.
# Запускается ВНУТРИ контейнера (от root), node.json лежит рядом.
#
# Зачем: с IP провайдера часть внешних API режется/отдаёт 403
# (проверено: api.groq.com → 403 с pve). Через Mihomo — нормальный доступ.
set -euo pipefail

NODE_JSON="${1:-/opt/furinya/node.json}"
MIX_PORT="${MIX_PORT:-7890}"
BIN=/usr/local/bin/mihomo
CONF_DIR=/etc/mihomo
CONF=$CONF_DIR/config.yaml

[ -f "$NODE_JSON" ] || { echo "Нет $NODE_JSON"; exit 1; }

echo "==> Ставим Mihomo"
apt-get update -qq
apt-get install -y -qq --no-install-recommends ca-certificates curl gzip jq

if [ ! -x "$BIN" ]; then
  echo "==> Качаем Mihomo"
  REL=$(curl -s https://api.github.com/repos/MetaCubeX/mihomo/releases/latest)
  TAG=$(echo "$REL" | jq -r .tag_name)
  # Имена ассетов: mihomo-linux-amd64-v1-<ver>.gz, mihomo-linux-amd64-compatible-<ver>.gz
  ASSET=""
  for pat in '^mihomo-linux-amd64-v1-.*\.gz$' '^mihomo-linux-amd64-compatible-.*\.gz$' '^mihomo-linux-amd64-.*\.gz$'; do
    A=$(echo "$REL" | jq -r '.assets[].name' | grep -E "$pat" | head -1)
    if [ -n "$A" ]; then ASSET="$A"; break; fi
  done
  [ -n "$ASSET" ] || { echo "не нашли linux-amd64 ассет в $TAG"; exit 1; }
  echo "    релиз $TAG, ассет $ASSET"
  url="https://github.com/MetaCubeX/mihomo/releases/download/${TAG}/${ASSET}"
  curl -fsSL -o /tmp/mihomo.gz "$url" || { echo "не скачался $url"; exit 1; }
  gunzip -c /tmp/mihomo.gz > "$BIN"
  chmod +x "$BIN"
fi
"$BIN" -v | head -1 || true

echo "==> Пишем конфиг Mihomo"
mkdir -p "$CONF_DIR"
SERVER=$(jq -r .server "$NODE_JSON")
PORT=$(jq -r .server_port "$NODE_JSON")
UUID=$(jq -r .uuid "$NODE_JSON")
SNI=$(jq -r '.tls.server_name // .server' "$NODE_JSON")
FP=$(jq -r '.tls.utls.fingerprint // "chrome"' "$NODE_JSON")
WS_PATH=$(jq -r '.transport.path // "/"' "$NODE_JSON")

cat > "$CONF" <<YAML
# Сгенерировано deploy/setup-mihomo.sh — прокси для внешних API бота.
mixed-port: ${MIX_PORT}
allow-lan: false
bind-address: 127.0.0.1
mode: rule
log-level: info
ipv6: false
unified-delay: true
tcp-concurrent: true

proxies:
  - name: vless-main
    type: vless
    server: ${SERVER}
    port: ${PORT}
    uuid: ${UUID}
    network: ws
    tls: true
    servername: ${SNI}
    client-fingerprint: ${FP}
    skip-cert-verify: false
    ws-opts:
      path: ${WS_PATH}
      headers:
        Host: ${SNI}

proxy-groups:
  - name: PROXY
    type: select
    proxies:
      - vless-main

rules:
  - GEOIP,private,DIRECT
  - MATCH,PROXY
YAML
chmod 600 "$CONF"

cat > /etc/systemd/system/mihomo.service <<'UNIT'
[Unit]
Description=Mihomo proxy client
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/mihomo -d /etc/mihomo
Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now mihomo
sleep 3

echo "==> Проверяем прокси"
for u in https://api.groq.com/openai/v1/models https://api.mistral.ai/v1/models https://api.fish.audio; do
  direct=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$u" 2>/dev/null || echo "—")
  proxied=$(curl -s -o /dev/null -w "%{http_code}" --max-time 15 -x "http://127.0.0.1:${MIX_PORT}" "$u" 2>/dev/null || echo "—")
  echo "    $u  напрямую=$direct  через mihomo=$proxied"
done
echo "==> Mihomo поднят на 127.0.0.1:${MIX_PORT}"
