//! Активный указатель памяти (data/diary/active.txt).
//!
//! Вместо файловой ротации (memory_1..5) мы держим ЕДИНСТВЕННУЮ память —
//! записи дневника `data/diary/<ts>.md` (ts = unix-секунды, служит ID и
//! ключом сортировки). «Активный контекст» = топ-K самых свежих НЕ-битых
//! записей. Если консолидация свежего таймстампа провалилась — он помечается
//! `corrupt`, и мы откатываемся к последней good-записи.
//!
//! `active.txt` — лёгкий индекс для O(1)-поиска активного без чтения всех
//! записей:
//! ```json
//! { "active_ts": "1784408283", "corrupt": ["1784408290", "1784408300"] }
//! ```
//! Поле `corrupt` дублирует флаг в метаданных записи (источник правды —
//! метаданные; active.txt — кэш/индекс).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::diary::Entry;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ActiveIndex {
    /// Самый свежий НЕ-битый таймстамп (id записи) — «активная» память.
    #[serde(default)]
    active_ts: String,
    /// Список таймстампов (id), помеченных битыми (corrupt).
    #[serde(default)]
    corrupt: Vec<String>,
}

/// Путь к active.txt относительно каталога дневника.
fn active_path(diary_dir: impl AsRef<Path>) -> PathBuf {
    diary_dir.as_ref().join("active.txt")
}

/// Читает индекс активного из active.txt (или дефолт, если нет/битый).
pub fn read_active(diary_dir: impl AsRef<Path>) -> (String, Vec<String>) {
    let path = active_path(diary_dir);
    match std::fs::read_to_string(&path) {
        Ok(s) => match serde_json::from_str::<ActiveIndex>(&s) {
            Ok(idx) => (idx.active_ts, idx.corrupt),
            Err(_) => (String::new(), Vec::new()),
        },
        Err(_) => (String::new(), Vec::new()),
    }
}

/// Пишет индекс активного в active.txt.
pub fn write_active(
    diary_dir: impl AsRef<Path>,
    active_ts: &str,
    corrupt: &[String],
) -> std::io::Result<()> {
    let idx = ActiveIndex {
        active_ts: active_ts.to_string(),
        corrupt: corrupt.to_vec(),
    };
    let json = serde_json::to_string_pretty(&idx)
        .unwrap_or_else(|_| "{\"active_ts\":\"\",\"corrupt\":[]}".to_string());
    std::fs::write(active_path(diary_dir), json)
}

/// Множество id битых записей (для защиты при prune).
pub fn corrupt_set(corrupt: &[String]) -> HashSet<String> {
    corrupt.iter().cloned().collect()
}

/// Возвращает активный контекст: топ-K самых свежих НЕ-битых записей
/// (склеенных по убыванию ts). `corrupt` — список id битых записей.
/// Если ни одной good-записи нет — возвращает пустой вектор.
pub fn active_context<'a>(entries: &'a [Entry], top_k: usize, corrupt: &[String]) -> Vec<&'a Entry> {
    let bad: HashSet<String> = corrupt_set(corrupt);
    let mut good: Vec<&Entry> = entries
        .iter()
        .filter(|e| !e.metadata.corrupt && !bad.contains(&e.id))
        .collect();
    // id — это unix-ts как строка фикс. длины → лексикографически == числовому.
    good.sort_by(|a, b| b.id.cmp(&a.id)); // свежие сверху
    good.into_iter().take(top_k).collect()
}

/// Самый свежий НЕ-битый таймстамп (активный), либо "" если нет.
pub fn active_ts(entries: &[Entry], corrupt: &[String]) -> String {
    active_context(entries, 1, corrupt)
        .first()
        .map(|e| e.id.clone())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diary::{Entry, EntryMetadata};

    fn entry(id: &str, corrupt: bool) -> Entry {
        Entry {
            id: id.to_string(),
            metadata: EntryMetadata {
                corrupt,
                ..Default::default()
            },
            body: format!("body-{}", id),
        }
    }

    #[test]
    fn active_context_picks_top_k_good_skips_corrupt() {
        let entries = vec![
            entry("100", false),
            entry("200", false),
            entry("300", true), // corrupt — должен быть исключён
            entry("400", false),
            entry("500", false),
        ];
        // top_k=2 → самые свежие good: 500, 400 (300 corrupt пропущен)
        let ctx = active_context(&entries, 2, &["300".to_string()]);
        let ids: Vec<&str> = ctx.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, vec!["500", "400"]);
    }

    #[test]
    fn active_context_empty_when_all_corrupt() {
        let entries = vec![entry("100", true), entry("200", true)];
        let ctx = active_context(&entries, 5, &["100".to_string(), "200".to_string()]);
        assert!(ctx.is_empty());
    }

    #[test]
    fn read_write_active_roundtrip() {
        let dir = std::env::temp_dir().join(format!("furinya_active_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        write_active(&dir, "12345", &["999".to_string()]).unwrap();
        let (ts, corrupt) = read_active(&dir);
        assert_eq!(ts, "12345");
        assert_eq!(corrupt, vec!["999".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

