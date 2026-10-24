# Деплой Furinya на Proxmox (LXC рядом с sd.cpp)

Всё делает **код** — бинарь `furinya-deploy` (никаких bash-скриптов):

```bash
cargo run --release -p furinya-deploy -- \
    --host 192.168.1.241 --key ~/.ssh/id_deploy_pve \
    --lxc-id 111 --lxc-ip 192.168.1.75
```

## Что делает деплой

1. **Собирает релиз** `cargo build --release -p furinya` (в `target-local/`, чтобы не
   мешать docker-сборкам, которые оставляют root-файлы в `target/`).
2. **Собирает payload**: бинарь, `libs/libtdjson.so` (830 МБ → 44 МБ после strip),
   `prompts/*.md`, `config.toml`, `data/tdlib_db` (сессия Telegram),
   `data/diary` (память), `.env` с ключами из env, systemd-юнит, клиент Mihomo
   (качается с GitHub) и его конфиг из `deploy/secrets/node.json`.
3. **По SSH (ssh2)**: создаёт LXC (если нет) на Debian 13, поднимает, ставит
   таймзону `Europe/Moscow`, ждёт сеть; заливает payload через SFTP →
   `pct push` → `pct exec` (распаковка в `/opt/furinya`).
4. **Ставит сервисы**: `mihomo.service` + `furinya.service` (systemd, `Restart=always`,
   `onboot=1` у LXC) и **проверяет**, что бот жив, печатая логи.

## Схема

```
Proxmox VE — 192.168.1.241
├── vmbr0 192.168.1.0/24  — интернет + sd-api
│   ├── LXC: sd-server (stable-diffusion.cpp) — 192.168.1.172:8082 (СТАТИКА)
│   └── LXC: furinya      — 192.168.1.75  (бот, systemd)
│         └── Mihomo 127.0.0.1:7890 — прокси для внешних API и Telegram
├── vmbr1 192.168.10.0/24 — без интернета (nginx-proxy, navidrome, …)
└── vmbr2 192.168.11.0/24 — изолированная песочница (honeypot)
```

Бот живёт на **vmbr0**: только там есть NAT в интернет и прямой доступ к sd-api.
В LXC — **Europe/Moscow** (тихие часы считаются по локальному времени юзера).

## Зачем Mihomo

С IP провайдера `api.groq.com` → **403** (Cloudflare режет), с домашней машины → 401.
Через Mihomo (VLESS-узел из `deploy/secrets/node.json`) → 200.
Telegram с провайдерского IP тоже заблокирован — поэтому TDLib ходит через
SOCKS5 на том же порту (`FURINYA_TG_PROXY=socks5://127.0.0.1:7890`).

## Эмбеддинги (Gemini)

Векторная память (RAG дневника, ситуационный поиск, извлечение фактов,
консолидация, семантическая часть анти-повтора) ходит на **Gemini
embeddings** (`gemini-embedding-001`) через OpenAI-совместимый слой Google:
`general.embedding.endpoint` в `config.toml`, ключ — `GEMINI_API_KEY` из env.

Локальный эмбеддер (llama.cpp + bge-m3) **удалён**: на этом хосте (Xeon без
AVX2) он считал запись ~4.6 с и грел железо, а облако отвечает ~1 с на пачку и
по реальным записям дневника держит релевантную запись в топ-1. У Gemini высокий
«фон» косинуса (~0.5 даже у несвязанных текстов), поэтому
`diary_min_relatedness` ≈ 0.55.

**Миграция векторов.** При смене эмбеддера старые векторы лежат в другом
векторном пространстве (даже при совпадении размерности 1024) и дают мусорные
«хиты». Бот при старте сравнивает `data/diary/.embed_model` (`baseUrl|model`)
с текущим эмбеддером и при расхождении **пересчитывает** дневник и
ситуационный индекс (см. `migrate_embeddings_if_needed` в
`bins/furinya/src/main.rs`). Маркер пишется только при успехе.

## Ключи и доступ

- API-ключи — **только из env** (`MISTRAL_API_KEY`, `GROQ_API_KEY`, `FISH_API_KEY`,
  `FURINYA_HORDE_KEY`); деплой переносит их в `/opt/furinya/.env` (chmod 600).
- SSH-ключ на Proxmox: `~/.ssh/id_deploy_pve` (по умолчанию), путь меняется `--key`.
- `deploy/secrets/node.json` — параметры VLESS-узла для Mihomo (в .gitignore).

## Аргументы деплоя

| Аргумент | По умолчанию | Смысл |
|---|---|---|
| `--host` | `192.168.1.241` | Proxmox-хост |
| `--lxc-id` | `111` | ID контейнера (101 занят navidrome) |
| `--lxc-ip` | `192.168.1.75` | адрес в LAN |
| `--bridge` | `vmbr0` | мост Proxmox |
| `--cores/--mem/--disk` | `2/2048/12` | ресурсы LXC |
| `--tz` | `Europe/Moscow` | таймзона в контейнере |
| `--no-build` | — | не пересобирать релиз |
| `--no-mihomo` | — | не ставить прокси |

## Логи и управление

```bash
ssh root@192.168.1.241 'pct exec 111 -- journalctl -u furinya -f'   # живой лог
ssh root@192.168.1.241 'pct exec 111 -- systemctl restart furinya'   # рестарт
```

## Забрать память после теста

```bash
ssh root@192.168.1.241 'pct exec 111 -- tar cf - -C /opt/furinya data/diary' \
  | tar xf - -C .
```

## Грабли, на которые уже наступили

| Проблема | Решение |
|---|---|
| `libtdjson.so` требует **GLIBC_2.38** | шаблон **debian-13** (glibc 2.41) |
| Локальная Void Linux (glibc 2.41) — бинарник пойдёт только туда же | собираем локально, LXC = Debian 13 (glibc 2.41) ✓ |
| `target/` после docker-сборок принадлежит root → cargo не может писать | отдельный `CARGO_TARGET_DIR=target-local` |
| Telegram-сессия одна на аккаунт | локальный бот держим выключенным |
| Ассет Mihomo называется `mihomo-linux-amd64-v1-<ver>.gz` | поиск по паттерну в API релиза |
| Шаблон ОС с хоста качается медленно | качаем заранее и заливаем через scp |
| `pveam download` требует `<storage> <template>` | `pveam download local debian-13-standard_*.tar.zst` |
