# furinya — Rust-порт kuni

LLM character AI (ru: Фуриня) на нативном Telegram user-account (TDLib 1.8.62).
Rust-миграция C++ проекта [kuni](https://github.com/Sanchoz2022/furinya) (история
kuni — в этом же репо, до思альных коммитов).

## Архитектура

```
crates/
  furinya-core    — конфиг, промпты, логгирование
  furinya-llm     — Mistral API клиент (chat, tools, мультимодальность)
  furinya-diary   — дневник-память (RAG: консолидация + embeddings Mistral)
  furinya-vision  — фото → текст (ministral-14b, декод по содержимому: WebP/PNG/JPEG)
  furinya-audio   — Groq STT (whisper-large-v3-turbo) + Fish Audio TTS (клон голоса)
  furinya-sd      — генерация фото: локальный sd-server (meinahentai, ~3.5 мин/фото)
                    и AI Horde fallback; NSFW-совместимо
  furinya-tg      — TDLib агенты: диалог, tools (сообщения/реакции/GIF/голос/фото),
                    живость (тайминги, склейки частей, опечатки-редакты, статусы)
bins/furinya      — main
prompts/          — личность и характер (НЕ в репо, локальные)
config.toml       — конфиги (НЕ в репо, см. config.example.toml)
libs/libtdjson.so — TDLib json-библиотека (НЕ в репо, 792MB)
```

## Запуск

```
cargo build --workspace
FURINYA_TD_QUIET=1 cargo run
```

ENV-ключи (никогда не в конфиге): `MISTRAL_API_KEY`, `GROQ_API_KEY`,
`FISH_API_KEY`, `FURINYA_HORDE_KEY`; опционально `SD_API_URL`.

## Спецификации площадки

- Нативный клиент TDLib → бот ведёт себя как обычный юзер (reactions, answers,
  voices, edit-fixes, статус online/offline).
- Mistral /text: `ministral-8b-2512` (625k TPM / 3.13 RPS) primary,
  `ministral-14b-2512` fallback и vision-multimodal.
- Голос: Fish TTS клон (v4), Groq STT; ГолС→ГС без текста в чате.
- Генерация фото — только локальный sd-server + LoRA furina/Hyper-SD
  (8 шагов, euler_a, clip_skip 2), Semaphore(1), NSFW по явному tool-флагу.
