//! Извлечение РЕАЛЬНЫХ фактов о собеседниках из per-чат выжимок
//! (`data/diary/chat_history/*.md`) в общую векторную память (diary).
//!
//! Это и есть «портрет юзера в RAG»: бот читает сжатые диалоги и вытягивает
//! имена, интересы, отношения, договорённости — как самодостаточные куски
//! памяти с тегами, готовые к embedding-kNN поиску.
//!
//! Не трогает TDLib. Работает офлайн на локальных файлах выжимок.

use anyhow::Result;
use furinya_core::prompts::Prompts;
use furinya_llm::{LlmClient, Message, MessageContent};
use std::path::Path;

use crate::diary::{Diary, Entry};

/// Сканирует все per-чат выжимки (`<diary_dir>/chat_history/*.md`), прогоняет
/// их через `diary_save`-промпт и сохраняет извлечённые факты в diary.
///
/// `keep_last` / `active_top_k` — параметры жёсткой ротации (обычно из
/// `cfg.misc`). После каждой записи старые файлы режутся до `keep_last`,
/// защищён активный контекст + corrupt-записи.
///
/// Возвращает количество реально сохранённых записей.
///
/// NOTE: промпт (`diary_save.md`) сам запрещает плодить «empty state»-мусор,
/// плюс `Diary::save_entries_from_text` имеет код-фильтр на такие фразы.
pub async fn build_profile_memory(
    llm: &LlmClient,
    diary: &Diary,
    prompts: &Prompts,
    diary_dir: &str,
    diary_entries: &mut Vec<Entry>,
    keep_last: usize,
    active_top_k: usize,
) -> Result<usize> {
    let chat_dir = Path::new(diary_dir).join("chat_history");
    let mut combined = String::new();
    if let Ok(entries) = std::fs::read_dir(&chat_dir) {
        for e in entries.flatten() {
            if let Ok(text) = std::fs::read_to_string(e.path()) {
                combined.push_str(&format!(
                    "\n--- {} ---\n{}\n",
                    e.file_name().to_string_lossy(),
                    text
                ));
            }
        }
    }
    if combined.trim().is_empty() {
        return Ok(0);
    }

    let system = format!(
        "{}\n\nИзвлеки из предоставленных чатов РЕАЛЬНЫЕ факты о собеседниках (имена, интересы, \
        отношения, настроение, договорённости). НЕ пиши 'пусто'/'empty state' — если фактов нет, \
        верни НИЧЕГО. Формат вывода — куски через '---', каждый с тегами #имя #отношения и т.п.",
        prompts.diary_save()
    );

    let resp = llm
        .chat(vec![Message::system(system), Message::user(combined)], None)
        .await?;
    let text = resp
        .choices
        .first()
        .and_then(|c| match &c.message.content {
            MessageContent::Text(s) => Some(s.clone()),
            MessageContent::Parts(_) => None,
        })
        .unwrap_or_default();

    let saved = diary
        .save_entries_from_text(&text, diary_entries, 50, keep_last, active_top_k)
        .await?;
    let n = saved.len();
    diary_entries.extend(saved);
    Ok(n)
}
