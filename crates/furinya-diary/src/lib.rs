//! furinya-diary — постоянная векторная память (RAG) для Furinya.
//!
//! Поведение взято из kuni (Diary.cpp / diary_save_entries.h):
//! хранение записей в `data/diary/<ts>.md`, эмбеддинги через bge-m3,
//! косинусный поиск, plagiarism-фильтр дубликатов.
//!
//! Единственная память = `data/diary/<ts>.md` (ts = unix-секунды, ID + ключ
//! сортировки). Активный указатель держится в `data/diary/active.txt`
//! (active_ts + corrupt[]). Файловая ротация (memory.md / memory_1..N.md)
//! удалена — см. Том 3.

pub mod diary;
pub mod embedding;
pub mod active;
pub mod consolidate;
pub mod memory_build;
pub mod situation;
pub use active::{active_context, active_ts, corrupt_set, read_active, write_active};
pub use consolidate::consolidate_memory;
pub use memory_build::build_profile_memory;
pub use situation::{SituationNote, SituationStore};

/// Одна реплика чата для консолидации.
///
/// В отличие от `furinya_llm::Message` (где есть только `role`+`content`),
/// здесь явно несём реальную дату и имя говорящего. Это нужно, чтобы
/// выжимка чата сохраняла хронологию и атрибуцию реплик (а не гадала
/// "кто есть кто" и не теряла даты, как было раньше).
#[derive(Debug, Clone)]
pub struct ChatTurn {
    /// Дата-время в читаемом виде, напр. "12.07 22:53".
    pub date: String,
    /// Имя говорящего (из config: character_name / papik_name), напр. "Фуринья" / имя из конфига.
    pub who: String,
    /// Текст реплики.
    pub text: String,
}

impl ChatTurn {
    /// Из сырых TDLib-данных `(sender_id, is_me, text, unix_date)`.
    /// `bot_name` / `user_name` берутся из config (character_name / papik_name).
    pub fn from_tg_full(
        rows: &[(i64, bool, String, i64)],
        bot_name: &str,
        user_name: &str,
    ) -> Vec<ChatTurn> {
        rows.iter()
            .map(|(_s, is_me, text, date)| ChatTurn {
                date: chrono::DateTime::from_timestamp(*date, 0)
                    .map(|d| d.format("%d.%m %H:%M").to_string())
                    .unwrap_or_else(|| "??.?? ??:??".to_string()),
                who: if *is_me {
                    bot_name.to_string()
                } else {
                    user_name.to_string()
                },
                text: text.clone(),
            })
            .collect()
    }

    /// Из `Vec<Message>` (только role+content, БЕЗ дат) — для точек, где история
    /// идёт из контекста диалога (без реальных таймстампов). Дата ставится "??.?? ??:??",
    /// имена — из config. Используется как fallback, если даты недоступны.
    pub fn from_messages_no_dates(
        msgs: &[furinya_llm::Message],
        bot_name: &str,
        user_name: &str,
    ) -> Vec<ChatTurn> {
        use furinya_llm::{MessageContent, Role};
        msgs.iter()
            .filter(|m| m.role != Role::System)
            .filter_map(|m| {
                let text = match &m.content {
                    MessageContent::Text(s) => s.clone(),
                    MessageContent::Parts(_) => String::new(),
                };
                let who = if m.role == Role::Assistant {
                    bot_name.to_string()
                } else {
                    user_name.to_string()
                };
                Some(ChatTurn {
                    date: "??.?? ??:??".to_string(),
                    who,
                    text,
                })
            })
            .collect()
    }
}

pub use diary::{Diary, Entry, EntryMetadata, QueryHit, cosine_similarity, parse_consolidated};
pub use embedding::EmbeddingClient;

