# Furinya — Roadmap / Статус разработки

> Сводка для продолжения работы в новом чате. Состояние на 2026-10-19.
> Проект: перенос C++ "kuni" → Rust "furinya" (TDLib user-account Telegram бот).
>
> ## !! ОБНОВЛЕНИЕ 2026-10-19: ЛОКАЛЬНЫЙ SD-СЕРВЕР + ЖИВЫЕ СООБЩЕНИЯ !!
> 1. **Генерация фото → наш sd-server** (stable-diffusion.cpp @ http://192.168.1.172:8082,
>    Proxmox LXC, CPU): чекпоинт `meinahentai` (SD1.5) + LoRA строго через корневой
>    массив "lora" (furina 0.75 SFW / 0.65 NSFW + Hyper-SD15-8steps 0.85), 8 шагов,
>    cfg 1.0, euler_a, clip_skip 2, 512×640, seed -1. Рендер ~3.5 мин.
>    Проверено live-пробами: SFW, NSFW (полный эксплицит без цензуры), аватар.
>    env `SD_API_URL` перекрывает config. Global Semaphore(1) — не дёргать CPU двумя
>    рендерами. Прогрев lora_cache (GET /sdapi/v1/loras) при старте и на ошибке lora.
>    Канон внешки — `prompts/character_appearance.md` (danbooru-теги Фурины),
>    инжектится в начало промпта. Название бэкенда в config: `take_photo.backend="sd"`.
>    NSFW-флаг: tool-параметр `nsfw` (только по явной просьбе) + авто-детект тегов промпта.
> 2. **Живость сообщений**: merge мелких частей (оба <40 симв → склейка до ~160),
>    максимум 6 (лучше 1-3); после merge отсылки идут с реалистичным набором текста:
>    send_action_typing + пауза 0.8–3.2с по длине части + джиттер 0-500мс, отбивка
>    250–800мс между сообщениями (было фиксированных 400мс).
> 3. **GIF**: tool `send_gif(query)` — встроенный инлайн-поиск Telegram (@gif,
>    searchPublicChat→getInlineQueryResults→sendInlineQueryResultMessage, БЕЗ API-ключей;
>    Giphy public key BANNED, Tenor v1 закрыт — поэтому TDLib inline). Уместные гифки
>    изредка; на голосовое — запрещено (guard, как send_telegram_message).
> 4. **Большой эмодзи**: одиночный эмодзи отдельной part → Telegram рендерит jumbo.
> 5. **TTS-фильтр**: ролевые действия в *звёздочках* вырезаются перед синтезом
>    (strip_asterisk_actions), плюс существовавшая чистка эмодзи/meta-строк.
> 6. **Аватар**: data/generated/furinya_avatar.png сгенерирован новым сервером и
>    УСТАНОВЛЕН как фото аккаунта (пример `cargo run -p furinya-tg --example set_avatar -- <png>`).
> 7. Vision/photo+caption fixes предыдущего дня: фото с подписью = подпись+описание в
>    контекст; WebP-декод по содержимому (Horde отдаёт WebP под именем .png);
>    vision = ministral-14b-2512 (8b — cheap); «сфоткайся» поддержа системным промптом.
>
> ## !! ОБНОВЛЕНИЕ 2026-10-17 (вечер): МИГРАЦИЯ СТЕКА НА ОБЛАЧНЫЕ API !!
> Стек переписан с локального самохостинга (Gemma/Whisper/SD @localhost) на облако:
> 1. **LLM**: Mistral API — основная `mistral-medium-latest`, резерв `ministral-14b-2512`.
>    **Live-проверено 2026-10-17**: `mistral-large-2512` НЕ доступен на бесплатном
>    тарифе (403 tier_not_allowed code 1910); medium/small/magistral/ministral —
>    доступны (список из GET /v1/models фильтруется по тарифу!).
>    RPM-лимит бесплатного тарифа ПО-МОДЕЛЬНЫЙ и жёсткий (~1-2 RPM: 429 code 1300):
>    ретрай 2с/4с (max 3 попытки) в send_with_rate_limit_retry, затем fallback на
>    ministral — он принимает и стрим, и native tool calls (проверено: react_with_emoji).
>    Багфикс: tool_choice шлётся ТОЛЬКО при наличии tools (иначе Mistral 400 3051).
>    Vision → `mistral-medium-latest` (multimodal; ministral текст-only).
>    Эмбеддинги (bge-m3 @8091) остались локальными — не тронуты.
>    **Ключи через ENV** (перекрывают пустые в config.toml): MISTRAL_API_KEY,
>    GROQ_API_KEY, FISH_API_KEY, FURINYA_HORDE_KEY (horde, env перегружает cfg.apikey).
>    Пустой env-ключ не портит чужой провайдер: mistral-fallback применяется только
>    когда baseUrl содержит api.mistral.ai.
> 2. **STT**: Groq Cloud `whisper-large-v3-turbo` (тот же OpenAI-совместимый
>    multipart — код не менялся, только endpoint в config). КРЕЙТ НЕ ИЗМЕНЁН.
> 3. **Картинки (NSFW)**: AI Horde (StableHorde) асинхронный протокол
>    async→check→status, r2=true, NEG через " ### " в промпте, всё в `furinya-sd` +
>    `[capabilities.take_photo.aihorde]`. Перекод любых webp/jpeg → PNG (Telegram).
>    **Live-проверено**: PNG записан и валиден за ~29с даже на анонимном ключе
>    (queue pos=0 сразу). Ловушки Horde: seed = СТРОКА ("-1"), булевы поля в
>    check приходят как 0/1 → tolerant_bool-десериализатор. Live-тест сохраняет
>    артефакт в data/generated/aihorde_test.png (не удаляет).
> 4. **TTS (голос Фурины)**: Fish Audio /v1/tts с referenceId (рус ОК), новый
>    tool `send_voice_message` + `TdClient::send_voice_note` (inputMessageVoiceNote).
>    **LIVE-ПРОВЕРЕНО 2026-10-18**: бесплатность работает ТОЛЬКО при передаче
>    model как HTTP-ЗАГОЛОВКА (`-H "model: s2.1-pro-free"`); JSON-поле model
>    НЕ спасает — API считает запрос платным (402 insufficient API credit).
>    Roundtrip Fish→Groq протестирован: RU-фраза синтезирована и распознана
>    слово в слово (examples/smoke_fish.rs, data/generated/fish_test.mp3).
>    FishTtsConfig::default() hand-written: пустая модель = платный откат,
>    поэтому default обязателен "s2.1-pro-free". reference_id опционален в
>    synthesize_fish_voice (дефолтный голос модели для тестов); агент же требует
>    клон-референс. Для ГОЛОСА Фурины нужен furina_ref.wav.
>    Гайд по клон-референсу: docs/voice-reference-guide.md.
>    РЕЗЕРВ-бэкенды TTS (если Fish упадёт): (а) локальный XTTS-v2 через
>    OpenAI-совместимый /audio/speech (backend="openai"); (б) Voxtral TTS
>    (voxtral-mini-tts-2603) — РУССКОГО НЕТ, код отложен.
>    **STT/Groq**: RU-регион Groq той танцует WAF-геоблок (403 без аргументов
>    даже для валидного ключа, cf-ray ARN) — нужен VPN/TUN включенный глобально.
>    С TUN — всё проходит (ключ валиден). env: GROQ_API_KEY.
> 5. КОНФИГ: ключи пустые по умолчанию — берутся из env (см. п.1). TOML-ловушка:
>    model ВНУТРИ endpoint.* (endpoint.model = "..."), не на уровне секции.
> 6. Тесты: cargo test --workspace зелёный (24 ok). Live (`-- --ignored`):
>    `HORDE_LIVE=1 cargo test -p furinya-sd -- --ignored --nocapture` (Horde),
>    `cargo test -p furinya-llm --test live_thinking -- --ignored --nocapture`
>    (Mistral стрим+tools; пропускается тихо, если нет ключа).
> 7. Консольный режим (main.rs) всё ещё опирается на kuni-форматы raw-парсинга
>    ("[Call tool: ...]"), для Mistral он бессмыслен, но безвреден (tools=None).

## Архитектура

Workspace (cargo), крейты:
- `furinya-core` — Config (TOML), Prompts (загрузка system/character/diary), logging (tracing→furinya.log + rotation), estimate_tokens.
- `furinya-llm` — OpenAI-совместимый клиент (chat_stream, extract_clean_reply, extract_thinking, parse_tool_calls).
- `furinya-diary` — EmbeddingClient (bge-m3 на :8091), Diary (RAG: query по cosine, save, load, tags, consolidate).
- `furinya-tg` — TDLib JSON FFI (worker-thread, TdClient: send_text, get_chats, send_action_typing, add_reaction; auth.rs: authenticate).
- `bins/furinya` — корневой бинарник (интерактивный чат + telegram_mode).

Запуск: `cargo run` (или `FURINYA_TD_QUIET=1 cargo run` чтобы скрыть TDLib-мусор в stderr).
Конфиг: `config.toml` (никогда не коммитить секреты: api_id/api_hash/telegram_my_id).

## Что СДЕЛАНО (проверено на реальном запуске)

### LLM ядро
- [x] OpenAI-совместимый streaming (chat_stream) на localhost:8090 (Gemma).
- [x] Интерактивный терминал: stdin → LLM → stdout, tool-loop (#ask мок), overflow-guard (20000 токенов).
- [x] extract_clean_reply (убирает теги tool-call из видимого текста).
- [x] extract_thinking (парсит ``).
- [x] !memory / !consolidate в терминале.

### Дневник / память (RAG)
- [x] Embeddings через bge-m3 (:8091).
- [x] cosine similarity query, теги (#Тег), фильтрация по relatedness (0.35).
- [x] consolidate_memory (sleep_consolidator: 12 записей → 2 консолидированные).
- [x] Дневник грузится из `/home/sanchoz2022/kuni/data/diary` (12 записей).

### TDLib Telegram (user-account)
- [x] Авторизация: flat setTdlibParameters (TDLib 1.8.62 требует плоскую структуру, НЕ вложенную "parameters"), waitPhoneNumber через stdin, waitEncryptionKey.
- [x] Worker-thread FFI (избегаем double-free: не free() на receive/execute результатах, Box::leak на lib).
- [x] Фетч чатов: getChats(100) → видим "Александр" (5005580127) + "Telegram" (777000).
- [x] updateNewMessage → LLM → sendMessage (дробный вывод).
- [x] my_id из config (telegram_my_id=8337379413) — устраняет петлю (не отвечаем себе).
- [x] send_action_typing (статус "печатает").
- [x] Дробный вывод: split_message по пустым строкам (`\n\n`), лимит max_message_parts=6.
- [x] Задержка между частями (WPM-имитация из config: typing_simulation_min/max_wpm).
- [x] add_reaction (addReaction с reactionTypeEmoji) — КОД ГОТОВ, но в рантайме не проверен (см. TODO).

### Логирование
- [x] tracing → furinya.log (file) + rotation (100MB / 5 files).
- [x] FURINYA_TD_QUIET=1 скрывает TDLib stderr-мусор.

## TODO (не закрыто, приоритет сверху)

### Блокеры рантайма
1. **Реакции не появляются** — `add_reaction` вызывается, но в Telegram реакция не ставится.
   Гипотезы: (a) TDLib 1.8.62 требует другой формат reaction; (b) message_id не тот
   (берём из `msg["id"]`, возможно нужен `msg["id"]` из updateNewMessage — проверить).
   Исправлено на `{ "@type": "reactionTypeEmoji", "reaction": emoji }` — но не проверено живьём.
2. **Thinking — black box** — модель пишет plain text БЕЗ `` блока, поэтому
   extract_thinking ничего не находит. В system.md добавлено требование писать
   `` всегда, но Gemma на :8090 его игнорирует (не обучена на такой формат?).
   Нужно: либо дообучить/настроить формат, либо использовать native thinking
   (если Gemma поддерживает), либо парсить структуру из plain text по маркерам.
   ВИДИМОСТЬ: в CLI thinking показывается жёлтым `\x1b[33m💭`, в TG — отдельным сообщением.

### Следующие шаги (после блокеров)
3. **Каналы/группы** — getChats уже фетчит все, но контент не читаем (28000-токен лимит).
4. **Фото через Gemma** (прямой вызов :8090 с base64, ~1300 токенов — проверено тобой).
5. **Аудио (whisper)** — отдельный шаг.
6. **Docker + mv моделей** из kuni в `models/`.
7. **ENV-переменные** в config (пока отложено).
8. **Thinking в TG** — решить, показывать ли 💭 сообщение пользователю (сейчас показываем).

## Структура файлов

```
furinya_td_ai_bot/
├── Cargo.toml                 (workspace)
├── config.toml                (секреты! не коммитить)
├── prompts/
│   ├── system.md              (инструкции: thinking, дробление, реакции)
│   ├── character_base.md
│   ├── character_appearance.md
│   ├── diary_save.md
│   └── sleep_consolidator.md
├── crates/
│   ├── furinya-core/src/{config,prompts,logging,lib}.rs
│   ├── furinya-llm/src/lib.rs
│   ├── furinya-diary/src/{embedding,diary,lib}.rs
│   └── furinya-tg/src/{lib,auth}.rs
├── bins/furinya/src/main.rs   (main + run_telegram_mode + split_message)
├── libs/libtdjson.so          (792MB, из kuni)
└── furinya.log                (rotation)
```

## Команды сборки/проверки
```
cargo fmt --all
cargo check -j 12
cargo clippy --all-targets -j 12
cargo test -j 12
FURINYA_TD_QUIET=1 cargo run
```

## Известные баги/нюансы
- TDLib 1.8.62: flat setTdlibParameters, send/receive в одном потоке (worker).
- getMe НЕ работает через наш канал (ответ не приходит) → my_id берём из config.
- rust-analyzer может показывать устаревшие ошибки (edition 2024) — проверять cargo check.
- Петля (Furinya отвечает сама себе) устранена через my_id фильтр.
- Дробление ТОЛЬКО по пустым строкам (маркеры `=====` и предложения убраны — ненадёжно).
