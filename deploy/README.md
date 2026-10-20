# Деплой Furinya на Proxmox (LXC рядом с sd.cpp)

Зачем: 2-дневный непрерывный прогон бота (память, инициатива, сон/тихие часы)
на always-on железе, а не на домашнем ПК, который спит и перезагружается.

## Схема

```
Proxmox VE — 192.168.1.241
├── vmbr0 192.168.1.0/24  — интернет + sd-api
│   ├── LXC: sd-server (stable-diffusion.cpp) — 192.168.1.172:8082 (СТАТИКА)
│   └── LXC: furinya      — 192.168.1.75  (бот, systemd)
│         └── Mihomo 127.0.0.1:7890 — прокси для внешних API
├── vmbr1 192.168.10.0/24 — без интернета (nginx-proxy, navidrome, …)
└── vmbr2 192.168.11.0/24 — изолированная песочница (honeypot)
```

Бот живёт на **vmbr0**: только там есть NAT в интернет и прямой доступ к sd-api.
В LXC — **Europe/Moscow** (тихие часы считаются по локальному времени юзера).

## Зачем Mihomo

С IP провайдера `api.groq.com` → **403** (Cloudflare режет), с домашней машины → 401.
Через Mihomo (VLESS-узел из локального v2rayN, wss + utls) → 200.
Бот ходит в API через `HTTP(S)_PROXY`, `NO_PROXY` исключает sd-сервер и локалку.
TDLib к Telegram идёт напрямую (env-прокси не читает).

## Особенности, на которые ушла кровь

| Проблема | Решение |
|---|---|
| `libtdjson.so` требует **GLIBC_2.38** | шаблон **debian-13** (glibc 2.41), не debian-12 |
| TDLib на свежей сессии отдаёт 1 сообщение вместо истории | `openChat` + `getChatHistory` + до 6 ретраев раз в 15с, пока в кэше < 20 |
| `searchChatMessages` в TDLib 1.8.62 отвечает `foundChatMessages`, а не `messages` | обрабатываются оба типа ответа |
| Ассет Mihomo называется `mihomo-linux-amd64-v1-<ver>.gz` | поиск по паттерну в API релиза |
| Шаблон ОС качается с хоста медленно (~50 КБ/с) | качать локально и заливать через scp |
| `pveam download` требует `<storage> <template>` | `pveam download local debian-13-standard_13.6-1_amd64.tar.zst` |

## Порядок

1. **Доступ**: публичный ключ в `/root/.ssh/authorized_keys` на Proxmox.
2. **Сборка** (на локальной машине): `./deploy/build-release.sh`
   — Docker-контейнер `debian:bookworm`, чтобы бинарник не зависел от
   glibc 2.41 локальной Void Linux.
3. **Деплой**: `./deploy/deploy.sh`
   — создаёт LXC, заливает бинарник + `libs/libtdjson.so` (830 МБ → 44 МБ после
   strip) + промпты + конфиг + `data/tdlib_db` (сессия Telegram) + `data/diary`
   (память), ставит systemd-юнит `furinya.service` + Mihomo, переносит ключи
   из env в `.env`.
4. **Локальный бот выключен** скриптом — аккаунт Telegram один, две сессии
   будут драться.

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
