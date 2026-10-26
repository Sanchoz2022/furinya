//! Консолидация памяти: жатие изменяемых записей дневника + пер-чат выжимка
//! истории + per-user профиль (слиты в один файл). Все промпты — ВНЕ кода
//! (prompts/*.md). Общей файловой памяти (memory.md / memory_1..N.md) больше
//! НЕТ — единственная память = data/diary/<ts>.md (ts = unix-секунды, ID +
//! ключ сортировки). Активный указатель держится в data/diary/active.txt.

use anyhow::Result;
use furinya_core::prompts::Prompts;
use furinya_llm::{LlmClient, Message, MessageContent};
use std::collections::HashSet;
use tracing::{info, warn};

use crate::{
    active::{active_context, corrupt_set, read_active, write_active},
    ChatTurn, Diary, Entry,
};

/// Консолидирует память:
/// 1. Жмёт изменяемые записи дневника (confidence < 1) в konsolidiert-записи.
/// 2. Жмёт историю чата И профиль юзера в ЕДИНЫЙ файл
///    `data/chat_history/<chat_id>.md` (профиль сверху, выжимка чата снизу).
/// 3. При неудачной консолидации дневника помечает свежую запись corrupt и
///    откатывается к последней good-записи (rollback).
/// 4. Обновляет active.txt (active_ts + corrupt[]) и перезагружает записи.
///
/// Возвращает количество сохранённых konsolidiert-записей.
#[allow(clippy::too_many_arguments)]
pub async fn consolidate_memory(
    llm: &LlmClient,
    diary: &Diary,
    prompts: &Prompts,
    diary_entries: &mut Vec<Entry>,
    diary_dir: &str,
    chat_history: &[ChatTurn],
    chat_id: i64,
    keep_last: usize,
    active_top_k: usize,
) -> Result<usize> {
    info!(
        "[consolidate] === НАЧАЛО КОНСОЛИДАЦИИ ПАМЯТИ (chat_id={}) ===",
        chat_id
    );
    let entries: &[Entry] = diary_entries;
    let mutables: Vec<&Entry> = diary.mutables(entries);

    // Читаем текущий active.txt (для отслеживания corrupt при rollback).
    let (_active_ts_prev, corrupt_prev) = read_active(diary_dir);

    // --- Шаг 1: жатие изменяемых записей дневника (опционально). ---
    let mut saved = Vec::new();
    let mut new_corrupt: Vec<String> = corrupt_prev.clone();
    if !mutables.is_empty() {
        let mut selected = mutables;
        selected.sort_by_key(|e| std::cmp::Reverse(e.id.clone()));
        selected.truncate(8);
        let mutables = selected;

        let mut input = String::new();
        for (i, e) in mutables.iter().enumerate() {
            if i > 0 {
                input.push_str("\n\n---\n\n");
            }
            input.push_str(&format!("{{\"confidence\":{}}}", e.metadata.confidence));
            input.push_str("\n\n");
            input.push_str(&e.body);
        }

        // Промпт — ИЗ ВНЕШНЕГО ФАЙЛА, не из кода.
        // Дата обязательна: без неё модель выдумывала год в «Тимestamps»
        // (26.10.2023 вместо 2026). Плюс явная граница ролей — иначе её
        // собственные idle-мысли превращались в «инициативу» собеседника.
        // Имя собеседника — из конфига/live-профиля, не из кода.
        let today = chrono::Local::now().format("%d.%m.%Y").to_string();
        let papik = prompts.papik_name();
        let system = format!(
            "{}\n\nСЕГОДНЯ: {today}. Все даты бери из входных записей и этого года — не выдумывай другой год. Если во входных записях НЕТ реплик второго участника ({papik}), не приписывай ему инициативу, действия, слова или ответы: это могли быть её собственные мысли и сообщения.\n\nCRITICAL: Output valid markdown ONLY. Each memory piece MUST start with a JSON object `{{\"confidence\":X}}` on its own line, followed by a blank line, then the body. Separate pieces with `\\n\\n---\\n\\n`. Do NOT wrap output in ```markdown``` fences. Do NOT add `**Rationale:**` or any meta-commentary outside the JSON+body format. If there is no real new information, output NOTHING.",
            prompts.sleep_consolidator()
        );

        let messages = vec![Message::system(system), Message::user(input.clone())];
        info!("[consolidate] запрос к LLM на жатие {} записей...", mutables.len());
        let resp = llm.chat(messages, None).await?;
        let reply = resp
            .choices
            .first()
            .and_then(|c| match &c.message.content {
                MessageContent::Text(s) => Some(s.clone()),
                MessageContent::Parts(_) => None,
            })
            .unwrap_or_default();

        let pieces = crate::parse_consolidated(&reply);
        if pieces.is_empty() {
            warn!("[consolidate] LLM не вернул валидных кусков (или пусто — ок)");
        } else {
            // Сохраняем. save_consolidated присваивает свежий ts (== id).
            saved = match diary.save_consolidated(&pieces, entries, 0, 0).await {
                Ok(s) => s,
                Err(e) => {
                    warn!("[consolidate] неудачная консолидация дневника: {:#}. Помечаем свежую запись corrupt → rollback.", e);
                    // Свежая запись — это самый новый id среди существующих
                    // (до reload). Помечаем corrupt и откатываемся.
                    if let Some(newest) = entries.iter().map(|e| e.id.clone()).max() {
                        let _ = diary.mark_corrupt(&newest);
                        if !new_corrupt.contains(&newest) {
                            new_corrupt.push(newest);
                        }
                    }
                    Vec::new()
                }
            };
            if !saved.is_empty() {
                info!(
                    "[consolidate] сохранено {} konsolidiert записей (было {})",
                    saved.len(),
                    mutables.len()
                );
            }
        }
    } else {
        info!("[consolidate] нет изменяемых записей — пропускаем diary-часть");
    }

    // --- Шаг 2: жатие истории чата + профиля юзера в ЕДИНЫЙ файл. ---
    // chat_history уже идёт хронологически (старые -> новые) и несёт (date, who, text).
    // Собираем ВЕСЬ чат без усечения — промпт сам решит, что сжать в "Ранее:", а что
    // оставить дословно. Раньше тут отбрасывались роли/даты -> выжимка теряла атрибуцию.
    let mut chat_text: String = String::new();
    for turn in chat_history {
        if turn.text.trim().is_empty() {
            continue;
        }
        chat_text.push_str(&format!("[{}] {}: {}\n", turn.date, turn.who, turn.text));
    }

    // Пропускаем, если реального диалога нет (не плодим "empty state").
    if chat_text.trim().len() < 50 {
        info!("[consolidate] диалог пустой/короткий (<50 симв) — пропускаем chat_history");
    } else {
        let (profile_old, history_old) = read_chat_memory(diary_dir, chat_id);
        let input = format!(
            "ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ (старый, сверху файла):\n{}\n\n---\n\nВЫЖИМКА ЧАТА (старая, снизу файла):\n{}\n\n---\n\nНОВЫЙ ДИАЛОГ ЗА СЕССИЮ (хронологически, старые сверху, формат [ДД.ММ ЧЧ:ММ] Имя: текст):\n{}\n\nОбнови ОБЕ ЧАСТИ: профиль (что Furinya думает о юзере) + выжимку чата. Верни ДВЕ секции, разделённые `---`.",
            profile_old, history_old, chat_text
        );
        let today = chrono::Local::now().format("%d.%m.%Y").to_string();
        let papik = prompts.papik_name();
        let system = format!(
            "{}\n\nСЕГОДНЯ: {today}. Год в датах — этот; не выдумывай другой. Не приписывай собеседнику ({papik}) слов или действий, которых нет в диалоге.",
            prompts.chat_history_consolidator()
        );
        let summary = llm
            .chat(vec![Message::system(system), Message::user(input)], None)
            .await?;
        let summary_text = summary
            .choices
            .first()
            .and_then(|c| match &c.message.content {
                MessageContent::Text(s) => Some(s.clone()),
                MessageContent::Parts(_) => None,
            })
            .unwrap_or_default();
        if !summary_text.trim().is_empty() {
            let dir = std::path::Path::new(diary_dir).join("chat_history");
            let _ = std::fs::create_dir_all(&dir);
            let path = dir.join(format!("{}.md", chat_id));
            let _ = std::fs::write(&path, summary_text);
            info!("[consolidate] per-чат выжимка+профиль сохранены: {:?}", path);
        }
    }

    // --- Шаг 2b: обновляем ЕЁ СОБСТВЕННОЕ состояние (self_state.md). ---
    // Per-чат профиль — про пользователя; этот файл — про НЕЁ саму (настроение,
    // чем занята, мысли). Он живёт между чатами и даёт модели материал, чтобы
    // писать первой своё живое, а не дежурное «ты тут?». Логика — в единой
    // функции update_self_state (её же зовёт стартовый бутстрап в furinya-tg).
    update_self_state(llm, prompts, diary_dir, &chat_text).await;

    // --- Шаг 3: обновляем active.txt (active_ts + corrupt[]) и reload. ---
    // Перезагружаем записи, чтобы pick-нуть свежий good-контекст.
    *diary_entries = diary.load_all().unwrap_or_default();
    let ctx = active_context(diary_entries, 1, &new_corrupt);
    let active_ts = ctx.first().map(|e| e.id.clone()).unwrap_or_default();
    write_active(diary_dir, &active_ts, &new_corrupt)?;
    info!(
        "[consolidate] active.txt: active_ts={}, corrupt={:?}",
        if active_ts.is_empty() {
            "<none>".to_string()
        } else {
            active_ts
        },
        new_corrupt
    );

    // --- Шаг 4: prune старых записей. ---
    // Защищаем: активный good-контекст (топ-K) + все помеченные corrupt
    // (нужны для rollback). Всё, что старше keep_last И не защищено — удаляем.
    let protected_ctx = active_context(diary_entries, active_top_k, &new_corrupt);
    let mut protected: HashSet<String> = corrupt_set(&new_corrupt);
    for e in &protected_ctx {
        protected.insert(e.id.clone());
    }
    let removed = diary.prune_old(diary_entries, keep_last, &protected)?;
    if removed > 0 {
        info!(
            "[consolidate] prune: удалено {} старых записей (keep_last={})",
            removed, keep_last
        );
        // Перезагружаем, чтобы вызывающий видел актуальное состояние диска.
        *diary_entries = diary.load_all().unwrap_or_default();
    }

    Ok(saved.len())
}

/// Маркеры «нужды»/привязки к собеседнику: если self_state их содержит —
/// состояние описывает ЕГО, а не её. Возвращает первое совпадение.
fn self_state_forbidden_hit(s: &str) -> Option<&'static str> {
    const BAD: &[&str] = &[
        "он ушёл",
        "пока он",
        "он не ответил",
        "он молчит",
        "ждёт его",
        "жду его",
        "его ответ",
        "он хочет",
        "он заметил",
        "мне одиноко",
        "недополуч",
        "чего-то не хватает",
        "скучаю",
        "грусть от того, что он",
        "без него",
        "с ним",
        "наши отношения",
        "отношения",
    ];
    let low = s.to_lowercase();
    BAD.iter().copied().find(|m| low.contains(m))
}

/// Обновляет `self_state.md` — ЕЁ СОБСТВЕННОЕ состояние (не про пользователя).
///
/// Единая точка входа для консолидации и стартового бутстрапа: держим файл
/// живым (настроение, чем занята, мысли), чтобы у модели был материал писать
/// первой своё, а не дежурное «ты тут?». Если модель уходит в «нужду» или
/// упоминает его/отношения — один повтор со строгой подсказкой.
///
/// `chat_text` — недавний диалог в формате `[ДД.ММ ЧЧ:ММ] Имя: текст`.
pub async fn update_self_state(
    llm: &LlmClient,
    prompts: &Prompts,
    diary_dir: &str,
    chat_text: &str,
) {
    let system0 = prompts.self_state();
    if system0.trim().is_empty() || chat_text.trim().len() < 50 {
        return;
    }
    let path = std::path::Path::new(diary_dir).join("self_state.md");
    let old_self = std::fs::read_to_string(&path).unwrap_or_default();
    let now = chrono::Local::now().format("%d.%m %H:%M").to_string();
    let input = format!(
        "ТЕКУЩЕЕ СОСТОЯНИЕ (старое, может быть пустым):\n{}\n\n---\n\nСЕЙЧАС: {}\n\n---\n\nНЕДАВНИЙ ДИАЛОГ (один чат, хронологически, старые сверху):\n{}\n\nОбнови её собственное состояние. Верни ТОЛЬКО файл.",
        if old_self.trim().is_empty() {
            "<пусто>".to_string()
        } else {
            old_self.trim().to_string()
        },
        now,
        chat_text
    );
    let mut system = system0;
    for attempt in 0..2 {
        match llm
            .chat(
                vec![Message::system(system.clone()), Message::user(input.clone())],
                None,
            )
            .await
        {
            Ok(resp) => {
                let body = resp
                    .choices
                    .first()
                    .and_then(|c| match &c.message.content {
                        MessageContent::Text(s) => Some(s.clone()),
                        MessageContent::Parts(_) => None,
                    })
                    .unwrap_or_default();
                if body.trim().is_empty() {
                    warn!("[self_state] пустой ответ LLM — файл не трогаю");
                    return;
                }
                if let Some(hit) = self_state_forbidden_hit(&body) {
                    if attempt == 0 {
                        warn!("[self_state] состояние про него («{hit}») — повтор со строгой подсказкой");
                        system.push_str(&format!(
                            "\n\nВАЖНО: прошлая попытка нарушила правила (найдено: «{hit}»). \
                             Пиши ТОЛЬКО про её собственный день. НИ ОДНОГО упоминания его, \
                             чата или отношений."
                        ));
                        continue;
                    }
                    warn!("[self_state] повтор снова с фреймом «{hit}» — принимаю как есть");
                }
                let _ = std::fs::write(&path, body.trim());
                info!("[self_state] обновлён: {:?}", path);
                return;
            }
            Err(e) => {
                warn!("[self_state] LLM ошибка: {e}");
                return;
            }
        }
    }
}

/// Читает старый per-чат файл (профиль + выжимка, слитые через `---`).
/// Возвращает (profile_text, history_text) — до/после первого разделителя.
fn read_chat_memory(diary_dir: &str, chat_id: i64) -> (String, String) {
    let path = std::path::Path::new(diary_dir)
        .join("chat_history")
        .join(format!("{}.md", chat_id));
    let content = std::fs::read_to_string(&path).unwrap_or_default();
    // Профиль — это всё до первого `\n---\n` (или первой строки `# Выжимка`).
    if let Some(idx) = content.find("\n---\n") {
        let (profile, history) = content.split_at(idx);
        (profile.trim().to_string(), history[4..].trim().to_string())
    } else {
        // Нет разделителя — целиком трактуем как профиль (совместимость со
        // старым форматом, где был только профиль в users/).
        (content.trim().to_string(), String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::self_state_forbidden_hit;

    #[test]
    fn self_state_guard_flags_needy_frames() {
        // Реальные «сползания» модели, которые guard должен ловить.
        assert!(self_state_forbidden_hit(
            "**Настроение:** лёгкая грусть от того, что он ушёл так быстро."
        )
        .is_some());
        assert!(self_state_forbidden_hit("мне одиноко без него").is_some());
        assert!(self_state_forbidden_hit("скучаю по его ответу").is_some());
        assert!(self_state_forbidden_hit("чего-то не хватает").is_some());
        assert!(self_state_forbidden_hit("думаю про наши отношения").is_some());
    }

    #[test]
    fn self_state_guard_passes_self_sufficient_state() {
        let good = "**Настроение:** спокойно и тепло, лёгкость после танца.\n\
                    **Чем занята:** устроилась у окна с тёплым чаем, листаю старые фото.\n\
                    **Мысли:** как хорошо, что у меня есть свои маленькие радости.";
        assert!(self_state_forbidden_hit(good).is_none());
        let good2 = "**Настроение:** тихо и ровно.\n\
                     **Чем занята:** разобрала свои вещи, повязала ленту на хвост.";
        assert!(self_state_forbidden_hit(good2).is_none());
    }
}
