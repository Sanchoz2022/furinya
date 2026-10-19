# Контекстное сжатие — для нового чата

> Этот файл — замена "памяти сессии". Если начинаешь новый чат с Hermes,
> скорми ему это + ROADMAP.md, и он продолжит без потери контекста.

## Кто ты (пользователь)
- Зовут "папик"/Саша, бот — "Фуринья" (Furinya). Соло-разработчик, мигрирует C++ "kuni" → Rust "furinya".
- Практичный, быстрый, низкая толерантность к абстракциям и фейковым результатам.
- Доверие заработано реальными прогонами (не println!-заглушками).
- Редактор: VS Code на Artix/CachyOS. rust-analyzer иногда врёт — проверяй cargo check.

## Что было сделано (кратко)
1. Вертикальный срез: терминал → LLM (streaming) → tool-loop.
2. Дневник RAG (embeddings bge-m3, cosine, consolidate).
3. Логирование + thinking в лог + overflow-guard (20000 токенов).
4. TDLib Telegram user-account: авторизация, фетч чатов, updateNewMessage → LLM → дробный sendMessage.
5. Дробный вывод (по пустым строкам, ≤6 частей), статус "печатает", реакции (код).
6. Устранена петля (my_id из config).

## Критические решения (не менять без обсуждения)
- TDLib ТОЛЬКО user-account (tdjson FFI), НЕ Bot API (требование иммерсии).
- Модели на localhost (:8090 Gemma chat, :8091 bge-m3 embeddings). Диск тесный — модели mv из kuni в `models/` позже.
- Перенос — REWRITE, не bit-for-bit. Поведение из kuni, не его грязный код.
- Дробление ТОЛЬКО по `\n\n` (маркеры/предложения убраны — ненадёжно).
- my_id = 8337379413 (Furinya), papik_chat_id = 5005580127 (ты). Оба в config.toml.

## Открытые баги (приоритет 1)
1. **Реакции не работают в рантайме** — addReaction вызывается, но реакция не появляется.
   Проверить: формат reactionTypeEmoji (уже исправлено), message_id из updateNewMessage.
2. **Thinking — black box** — Gemma не пишет `` блок, extract_thinking пуст.
   Нужно либо заставить модель писать структуру, либо парсить из plain text.
   Видимость thinking в CLI (жёлтый) и TG (💭 сообщение) готова, но данных нет.

## Что делать завтра
1. Починить реакции (дебаг TDLib лог: `timeout 15 ./target/debug/furinya 2>&1 | grep -i reaction`).
2. Починить thinking (либо формат Gemma, либо парсинг plain text по `thought:`/`emotion:`).
3. Проверить живьём дробный вывод (напиши Furinya в TG, посмотри лог).
4. Каналы/группы (deferred — 28000 токенов лимит).
5. Фото через Gemma (прямой base64 вызов, ~1300 токенов).
6. Docker + mv моделей.

## Команды
```
cargo fmt --all && cargo check -j 12 && cargo clippy --all-targets -j 12
FURINYA_TD_QUIET=1 cargo run
git log --oneline -10   # история коммитов
cat ROADMAP.md          # полный статус
```

## Секреты (НЕ КОММИТИТЬ)
- config.toml: telegram_api_id=26039254, telegram_api_hash=[REDACTED], telegram_my_id=8337379413, papik_chat_id=5005580127.
- libtdjson.so (792MB) — локально, не в git.
- Дневник: /home/sanchoz2022/kuni/data/diary (12 записей, НЕ трогать оригинал).
