# Furinya — Контекстный мап (дополнение к grill-me.md)

> Отдельный файл для быстрой навигации: ГДЕ что лежит, ЧТО сделано, КАК проверить.
> grill-me.md — хронология (Тома). Этот файл — snapshot структуры на 2026-07-19.

## Структура файлов (что реально на диске)
```
crates/furinya-llm/src/lib.rs      — LLM streaming + tool_choice:"required" + thinking
crates/furinya-core/src/prompts.rs — build_system_prompt (+epilogue/anti_repeat), аксессоры
crates/furinya-diary/src/consolidate.rs — consolidate_memory (ротация, промпты вне кода)
crates/furinya-tg/src/agent.rs     — TG-цикл + stdin-команды + handle_new_message
bins/furinya/src/main.rs            — точка входа (TG-режим return run_telegram_mode)
prompts/system.md                  — системный промпт (gitignored, на диске)
prompts/chat_history_consolidator.md — verbatim диалог + эмоции чата
prompts/user_consolidator.md       — per-user профиль
prompts/sleep_consolidator.md      — общая память (ДОПОЛНЯТЬ, не стирать)
prompts/diary_save.md              — запрет пустых записей
data/diary/*.md                    — 16 diary-записей (большинство старые "empty state" из kuni)
data/diary/memory_1.md, memory_2.md — КОСТЫЛЬ (нумерация, юзер сказал "убрать")
data/diary/memory.md               — СТАРЫЙ, мусорит (legacy "Карта памяти", НЕ удаляется)
data/diary/chat_history/5005580127.md — per-chat выжимка (есть, но не обновляется при консоле)
data/diary/users/                  — ПУСТО (баг: не создаётся)
data/diary/debug/                  — ПУСТО (raw JSON не пишется)
```

## Коммиты (последние)
- `1280f4e` — команды только в stdin, memory build извлекает реальные факты
- `b3e3968` — 3 сырых блока диагностики (THINK/REPLY/TOOL) + промпты вне кода + ротация
- `4885db0` — строго tool-call ответы, без утечки раздумий в чат

## Как проверить (для пользователя)
1. Запуск (интерактивно, НЕ через `| tee` — иначе stdin закрыт):
   ```
   cd /home/sanchoz2022/furinya_td_ai_bot
   FURINYA_TD_QUIET=1 cargo run
   ```
2. В TG писать "Ты тут?" — модель отвечает (THINK/REPLY/TOOL в логе).
3. В терминале (РЕЖИМ КОМАНД): `memory build` → извлекает факты в data/diary/*.md.
4. `consolidate` / `save` → сжатие памяти. `exit` → выход.

## Баги (кратко)
- chat_history не обновляется при консоле (tg_history_full пуст при вызове).
- users/ не создаётся (тот же баг).
- memory.md legacy не удаляется (rotate_memory не трогает).
- debug/ пустой (raw JSON не пишется).

## Архитектура ротации памяти (НОВАЯ, из Том 3)
> Юзер вынес СЕГОДНЯ. Решает проблему контекста (белка в колесе из kuni: 1 запрос = 11100 токенов).
- В контексте висят ТОЛЬКО: **системный промпт + долгосрочная память** (memory.md — основные
  доки ~5 шт; + chat-доки + user-доки по чату).
- **НЕ подгружаем слой контекста перед КАЖДЫМ запросом** (иначе 2 запроса = вся память в трубу).
- По ходу чата LLM догружает детали через рассуждение (RAG `#ask`).
- При **context-лимите** → компрессия (consolidate) → снова в память. Круг.
- **Файлы**: memory.md (основные ~5) + chat-доки + user-доки. Можно слить (близкие векторы).
- **КАТЕГОРИЧЕСКИ НЕТ**: нумерации memory_1..5 (обоссанная куни-практика). Ротация =
  контекстная подгрузка, НЕ файловая нумерация.

## Что делать завтра (приоритет)
1. ЧИСТКА: убрать memory_1..5 (rotate_memory переписать), удалить legacy memory.md.
2. Разделить "3 в 1" на 3 слоя (общая / per-chat / per-user) по новой архитектуре.
3. Чиним баг chat_history/users (собирать chat_msgs из tg_history).
4. (потом) Media (приоритет 3).
