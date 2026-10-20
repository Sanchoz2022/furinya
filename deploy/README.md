# Деплой Furinya на Proxmox (LXC рядом с sd.cpp)

Зачем: 2-дневный непрерывный прогон бота (память, инициатива, сон/тихие часы)
на always-on железе, а не на домашнем ПК, который спит и перезагружается.

## Схема

```
Proxmox VE — 192.168.1.241
├── LXC: sd-server (stable-diffusion.cpp) — 192.168.1.172:8082   ← генерация фото
└── LXC: furinya      — 192.168.1.75                             ← бот (создаёт deploy.sh)
        └── SD_API_URL=http://192.168.1.172:8082/
```

## Требования на хосте

- Proxmox VE 8/9, шаблон `debian-12` (скачивается автоматически);
- мост `vmbr0` в LAN (иначе правится `LXC_BRIDGE`);
- свободный IP в LAN для LXC (`LXC_IP`).

## Порядок

1. **Доступ**: добавить публичный ключ в `/root/.ssh/authorized_keys` на Proxmox.
2. **Сборка** (на локальной машине): `./deploy/build-release.sh`
   — Docker-контейнер `debian:bookworm`, чтобы бинарник не зависел от
   glibc 2.41 локальной Void Linux.
3. **Деплой**: `./deploy/deploy.sh`
   — создаёт LXC, заливает бинарник + `libs/libtdjson.so` (830 МБ!) + промпты +
   конфиг + `data/tdlib_db` (сессия Telegram) + `data/diary` (память),
   ставит systemd-юнит `furinya.service`, переносит ключи из env в `.env`.
4. **Локальный бот выключить** — аккаунт Telegram один, две сессии будут драться.

## Переменные

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `PVE_HOST` | `192.168.1.241` | Proxmox-хост |
| `LXC_ID` | `101` | ID контейнера |
| `LXC_IP` | `192.168.1.75` | адрес в LAN |
| `LXC_BRIDGE` | `vmbr0` | мост Proxmox |
| `LXC_CORES/MEM/DISK` | `2/2048/8` | ресурсы |
| `LXC_TEMPLATE` | `local:vztmpl/debian-12-standard_12.12-1_amd64.tar.zst` | шаблон ОС |

## Логи и управление

```bash
ssh root@192.168.1.241 'pct exec 101 -- journalctl -u furinya -f'   # живой лог
ssh root@192.168.1.241 'pct exec 101 -- systemctl restart furinya'   # рестарт
```

## Обратно на локальную машину (собрать память теста)

```bash
scp -r root@192.168.1.241:/var/lib/vz/... # или
ssh root@192.168.1.241 'pct exec 101 -- tar czf - -C /opt/furinya data/diary' \
  | tar xzf - -C .
```

## Замечания

- `libtdjson.so` — 830 МБ, положить придётся (сборка TDLib на месте — часы);
  можно предварительно `strip --strip-debug` (минус ~40%).
- Telegram-сессия лежит в `data/tdlib_db` — копируется вместе с payload,
  повторный вход по коду не нужен.
- Пути: `libs/libtdjson.so` ищется относительно рабочего каталога
  (`WorkingDirectory=/opt/furinya` в юните).
