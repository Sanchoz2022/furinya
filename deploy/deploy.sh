#!/usr/bin/env bash
# Деплой Furinya в отдельный LXC на Proxmox-хосте.
#
# Что делает:
#   1. создаёт LXC (Debian 12) с адресом в LAN — рядом с sd.cpp;
#   2. заливает payload: бинарник, libs/libtdjson.so, prompts/, config.toml,
#      data/tdlib_db (сессия Telegram), data/diary (память);
#   3. ставит systemd-юнит + .env с ключами и запускает.
#
# Запуск (с локальной машины, где есть ssh-доступ к Proxmox):
#   ./deploy/deploy.sh
#
# Переменные (можно переопределить):
#   PVE_HOST=192.168.1.241 LXC_ID=101 LXC_IP=192.168.1.75
set -euo pipefail
cd "$(dirname "$0")/.."

PVE_HOST="${PVE_HOST:-192.168.1.241}"
LXC_ID="${LXC_ID:-111}"          # 101 занят navidrome
LXC_NAME="${LXC_NAME:-furinya}"
LXC_IP="${LXC_IP:-192.168.1.75}"  # vmbr0 — единственный мост с интернетом и доступом к sd-api
LXC_CIDR="${LXC_CIDR:-24}"
LXC_GW="${LXC_GW:-192.168.1.1}"
LXC_BRIDGE="${LXC_BRIDGE:-vmbr0}"
LXC_CORES="${LXC_CORES:-2}"
LXC_MEM="${LXC_MEM:-2048}"
LXC_DISK="${LXC_DISK:-12}"
SD_LXC_ID="${SD_LXC_ID:-110}"      # sd-api
MIX_PORT="${MIX_PORT:-7890}"
APP_DIR="/opt/furinya"

ssh_pve() { ssh -o BatchMode=yes -o ConnectTimeout=10 "root@$PVE_HOST" "$@"; }

echo "==> Проверка доступа к $PVE_HOST"
ssh_pve 'hostname; pveversion | head -1'

# ── 1. Шаблон ОС ────────────────────────────────────────────────────────────
echo "==> Ищем шаблон Debian 12"
LXC_TEMPLATE="$(ssh_pve "pvesm list local --content vztmpl | awk '/debian-12/ {print \$1}' | head -1")"
if [ -z "$LXC_TEMPLATE" ]; then
  echo "==> Скачиваем шаблон Debian 12"
  ssh_pve 'pveam update && pveam download debian-12 standard'
  LXC_TEMPLATE="$(ssh_pve "pvesm list local --content vztmpl | awk '/debian-12/ {print \$1}' | head -1")"
fi
[ -n "$LXC_TEMPLATE" ] || { echo "Шаблон не найден"; exit 1; }
echo "    шаблон: $LXC_TEMPLATE"

# ── 2. Создание/старт LXC ──────────────────────────────────────────────────
if ! ssh_pve "pct status $LXC_ID" >/dev/null 2>&1; then
  echo "==> Создаём LXC $LXC_ID ($LXC_NAME) на $LXC_IP"
  # Storage для rootfs: пробуем LVM → dir → zfs (что есть на хосте).
  CREATED=0
  for ST in "${LXC_STORAGE:-local-lvm}" local local-zfs; do
    if ssh_pve "pct create $LXC_ID $LXC_TEMPLATE \
      --hostname $LXC_NAME --cores $LXC_CORES --memory $LXC_MEM --swap 512 \
      --rootfs ${ST}:${LXC_DISK} --ostype debian --unprivileged 1 \
      --net0 name=eth0,bridge=$LXC_BRIDGE,ip=$LXC_IP/$LXC_CIDR,gw=$LXC_GW \
      --onboot 1" 2>&1 | tail -3; then
      if ssh_pve "pct status $LXC_ID" >/dev/null 2>&1; then
        echo "    rootfs на storage: $ST"; CREATED=1; break
      fi
    fi
  done
  [ "$CREATED" = "1" ] || { echo "Не удалось создать LXC"; exit 1; }
  echo "==> Старт LXC"
  ssh_pve "pct start $LXC_ID"
  echo "==> Ставим базу (обновление apt — заодно проверка сети)"
  ssh_pve "pct exec $LXC_ID -- bash -c 'apt-get update -qq && apt-get install -y -qq ca-certificates tzdata'"
  # Таймзона: тихие часы считаются по ЛОКАЛЬНОМУ времени (у юзера MSK),
  # в LXC по умолчанию UTC — иначе окно сна съедет на 3 часа.
  ssh_pve "pct exec $LXC_ID -- bash -c 'ln -sf /usr/share/zoneinfo/Europe/Moscow /etc/localtime && echo Europe/Moscow > /etc/timezone && date'"
else
  echo "==> LXC $LXC_ID уже существует"
  ssh_pve "pct start $LXC_ID" 2>/dev/null || true
  sleep 3
fi

# ── 3. Payload ─────────────────────────────────────────────────────────────
echo "==> Готовим payload"
[ -x target/deploy/furinya ] || { echo "Нет target/deploy/furinya — сначала ./deploy/build-release.sh"; exit 1; }

# Локальный бот держит ту же Telegram-сессию — останавливаем, чтобы копия
# data/tdlib_db была консистентной (SQLite WAL) и чтобы не драться за аккаунт.
if pgrep -f 'target/(debug|release)/furinya' >/dev/null 2>&1; then
  echo "==> Останавливаем локальный бот (одна Telegram-сессия на аккаунт)"
  pkill -f 'target/(debug|release)/furinya' || true
  sleep 3
fi

STAGE="$(mktemp -d)"
mkdir -p "$STAGE/payload/libs" "$STAGE/payload/prompts" "$STAGE/payload/data"
cp target/deploy/furinya            "$STAGE/payload/furinya"
cp libs/libtdjson.so                "$STAGE/payload/libs/"
# .so весит 830 МБ — выкидываем отладочные символы (на функционал не влияет)
strip --strip-debug "$STAGE/payload/libs/libtdjson.so" 2>/dev/null || true
echo "    libtdjson.so: $(du -h "$STAGE/payload/libs/libtdjson.so" | cut -f1) (было 830M)"
cp prompts/*.md                     "$STAGE/payload/prompts/"
cp config.toml                      "$STAGE/payload/config.toml"
cp -r data/tdlib_db/.               "$STAGE/payload/data/tdlib_db/" 2>/dev/null || true
cp -r data/diary/.                  "$STAGE/payload/data/diary/" 2>/dev/null || true
cp deploy/setup-mihomo.sh           "$STAGE/payload/setup-mihomo.sh"
cp deploy/secrets/node.json         "$STAGE/payload/node.json" 2>/dev/null || true

# Адрес sd-api: он на DHCP в vmbr0 — узнаём актуальный и пишем в .env,
# чтобы sd-сервер не уезжал.
SD_IP="$(ssh_pve "pct exec $SD_LXC_ID -- hostname -I" 2>/dev/null | awk '{print $1}')"
echo "    sd-api ($SD_LXC_ID) на $SD_IP"
if [ -z "$SD_IP" ]; then
  SD_IP="$(ssh_pve "pvesh get /nodes/$(hostname)/lxc/$SD_LXC_ID/interface/0/ip 2>/dev/null" | tr -d '"')"
  SD_IP="${SD_IP%%/*}"
fi

# .env с ключами из текущего окружения (значения в лог не печатаются)
{
  echo "MISTRAL_API_KEY=${MISTRAL_API_KEY:-}"
  echo "GROQ_API_KEY=${GROQ_API_KEY:-}"
  echo "FISH_API_KEY=${FISH_API_KEY:-}"
  echo "FURINYA_HORDE_KEY=${FURINYA_HORDE_KEY:-}"
  echo "SD_API_URL=${SD_API_URL:-http://${SD_IP:-192.168.1.172}:8082/}"
  # Прокси через Mihomo (см. deploy/setup-mihomo.sh): без него api.groq.com → 403.
  # NO_PROXY — чтобы sd-сервер и локалка шли напрямую, мимо прокси.
  echo "HTTP_PROXY=http://127.0.0.1:${MIX_PORT}"
  echo "HTTPS_PROXY=http://127.0.0.1:${MIX_PORT}"
  echo "NO_PROXY=127.0.0.1,localhost,${SD_IP:-192.168.1.172},192.168.1.0/24,10.0.0.0/8"
  echo "FURINYA_TD_QUIET=1"
} > "$STAGE/payload/.env"

echo "==> Заливаем в LXC ($(du -sh "$STAGE/payload" | cut -f1))"
tar -C "$STAGE/payload" -czf - . | ssh_pve \
  "pct exec $LXC_ID -- bash -c 'rm -rf $APP_DIR && mkdir -p $APP_DIR && tar xzf - -C $APP_DIR'"

# ── 4. Пользователь, systemd ──────────────────────────────────────────────
echo "==> Настраиваем сервис"
scp -q deploy/furinya.service "root@$PVE_HOST:/tmp/furinya.service"
ssh_pve "pct push $LXC_ID /tmp/furinya.service /etc/systemd/system/furinya.service"
# ── Mihomo (прокси для внешних API) ────────────────────────────────────────
if [ -f "$STAGE/payload/node.json" ]; then
  echo "==> Ставим Mihomo в LXC"
  ssh_pve "pct exec $LXC_ID -- bash /opt/furinya/setup-mihomo.sh /opt/furinya/node.json" || \
    echo "    ⚠ Mihomo не поставился — бот будет ходить напрямую"
fi
ssh_pve "pct exec $LXC_ID -- bash -c '
  set -e
  id furinya >/dev/null 2>&1 || useradd -m -s /bin/bash furinya
  chown -R furinya:furinya $APP_DIR
  chmod 700 $APP_DIR/data/tdlib_db || true
  chmod 600 $APP_DIR/.env
  systemctl daemon-reload
  systemctl enable --now furinya
'"
sleep 8
ssh_pve "pct exec $LXC_ID -- bash -c 'systemctl --no-pager --lines=30 status furinya || journalctl -u furinya -n 40 --no-pager'"

rm -rf "$STAGE"
echo
echo "==> Готово. Логи: ssh root@$PVE_HOST 'pct exec $LXC_ID -- journalctl -u furinya -f'"
echo "==> Локальный бот остановлен — он и не должен работать параллельно (аккаунт один)."
