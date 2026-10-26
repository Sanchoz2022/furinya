//! Внутренняя оценка хода (assessment) — «размышление» перед ответом.
//!
//! kuni получал оценку состояния/намерения собеседника прямо в тексте ответа
//! (`- thought:` / `- emotion:`), а Rust-порт ждал native `reasoning_content`,
//! которого `ministral-8b-2512` не отдаёт. Из-за этого этап оценки просто
//! исчез, и бот отвечал generic-заглушками.
//!
//! Поэтому оценка вынесена в ОТДЕЛЬНЫЙ дешёвый LLM-вызов (temperature 0.3,
//! без tools): модель возвращает ровно три строки —
//!   `СОСТОЯНИЕ ОН:` / `СОСТОЯНИЕ Я:` / `ПЛАН:`
//! — которые инжектятся в system-промпт хода («Оценка хода») и пишутся в
//! per-chat файл состояния `data/diary/chat_state/<chat_id>.md`.
//!
//! Per-chat состояние — про его и моё эмоциональное состояние между ходами;
//! общий `self_state.md` (про её собственный день) — отдельная сущность.

use std::path::{Path, PathBuf};

/// Разобранная оценка хода.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Assessment {
    /// Его состояние и что он хочет.
    pub he: String,
    /// Моё состояние.
    pub me: String,
    /// План ответа (что делать и чего НЕ делать).
    pub plan: String,
}

/// Приводит строку к виду `label: value` — снимает markdown-жирность, маркеры
/// списка, нумерацию и разные разделители (`:` / `—` / `-`).
fn split_label(line: &str) -> Option<(&str, &str)> {
    let mut s = line.trim();
    // Снимаем ведущие маркеры: "- ", "* ", "1. ", "• ".
    loop {
        let before = s;
        s = s.trim_start();
        if let Some(rest) = s.strip_prefix("- ") {
            s = rest;
        } else if let Some(rest) = s.strip_prefix("* ") {
            s = rest;
        } else if let Some(rest) = s.strip_prefix("• ") {
            s = rest;
        } else {
            let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
            if !digits.is_empty() && s[digits.len()..].starts_with(". ") {
                s = &s[digits.len() + 2..];
            } else {
                break;
            }
        }
        if s == before {
            break;
        }
    }
    // Снимаем markdown-жирность/курсив по краям метки.
    let s = s.trim_start_matches('*').trim_start();
    let colon = s.find(':')?;
    let label = s[..colon].trim().trim_matches('*').trim();
    let value = s[colon + 1..].trim().trim_matches('*').trim();
    Some((label, value))
}

/// Разбирает ответ модели оценки. Терпим к регистру, markdown-жирности,
/// маркерам списка и нумерации. `None` — если не нашлось хотя бы его
/// состояния и плана (тогда оценка не используется вовсе).
pub fn parse_assessment(raw: &str) -> Option<Assessment> {
    let mut he = String::new();
    let mut me = String::new();
    let mut plan = String::new();
    for line in raw.lines() {
        let Some((label, value)) = split_label(line) else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        let label_up = label.to_uppercase();
        if label_up.starts_with("СОСТОЯНИЕ ОН") {
            he = value.to_string();
        } else if label_up.starts_with("СОСТОЯНИЕ Я") {
            me = value.to_string();
        } else if label_up.starts_with("ПЛАН") {
            plan = value.to_string();
        }
    }
    if he.is_empty() || plan.is_empty() {
        return None;
    }
    if me.is_empty() {
        me = "—".to_string();
    }
    Some(Assessment { he, me, plan })
}

/// Блок, который инжектится в system-промпт хода.
pub fn format_assessment(a: &Assessment) -> String {
    format!(
        "# Оценка хода (внутренняя)\n\
         ЭТО ТВОЙ ВНУТРЕННИЙ РАЗБОР ТЕКУЩЕГО ХОДА. Отвечай в согласии с ним: он — не текст для чата,\n\
         его нельзя цитировать или пересказывать. План — это направление и тон, а не готовый текст\n\
         ответа: отзовись на него живым голосом и только потом задай вопрос, если он уместен.\n\
         СОСТОЯНИЕ ОН: {}\n\
         СОСТОЯНИЕ Я: {}\n\
         ПЛАН: {}",
        a.he, a.me, a.plan
    )
}

/// Путь per-chat файла состояния: `<diary_dir>/chat_state/<chat_id>.md`.
pub fn chat_state_path(diary_dir: &Path, chat_id: i64) -> PathBuf {
    diary_dir.join("chat_state").join(format!("{chat_id}.md"))
}

/// Читает прошлый per-chat state (пустая строка, если файла нет).
pub fn read_chat_state(diary_dir: &Path, chat_id: i64) -> String {
    std::fs::read_to_string(chat_state_path(diary_dir, chat_id)).unwrap_or_default()
}

/// Пишет per-chat состояние «моё состояние с ним».
pub fn write_chat_state(
    diary_dir: &Path,
    chat_id: i64,
    a: &Assessment,
    updated: &str,
) -> std::io::Result<()> {
    let path = chat_state_path(diary_dir, chat_id);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let body = format!(
        "# Моё состояние с ним\n\nОн: {}\nЯ: {}\nОбновлено: {}\n",
        a.he, a.me, updated
    );
    std::fs::write(path, body)
}

/// Собирает вход для LLM-вызова оценки: его последнее сообщение + хвост
/// истории + прошлая оценка (для непрерывности).
pub fn build_assess_input(last_message: &str, history_tail: &str, prev_state: &str) -> String {
    let mut s = String::new();
    if !prev_state.trim().is_empty() {
        s.push_str("ПРЕДЫДУЩАЯ ОЦЕНКА (для непрерывности, может быть пустой):\n");
        s.push_str(prev_state.trim());
        s.push_str("\n\n");
    }
    if !history_tail.trim().is_empty() {
        s.push_str("НЕДАВНИЙ ЧАТ (хронологически, старые сверху):\n");
        s.push_str(history_tail.trim());
        s.push_str("\n\n");
    }
    s.push_str("ЕГО ПОСЛЕДНЕЕ СООБЩЕНИЕ (оцени именно его):\n");
    s.push_str(last_message.trim());
    s.push_str("\n\nВерни ровно три строки: СОСТОЯНИЕ ОН:, СОСТОЯНИЕ Я:, ПЛАН:");
    s
}

/// Вход оценки для ВНУТРЕННЕГО хода (idle): его сообщения нет вообще.
///
/// Раньше на idle оценка не запускалась, и инициативные сообщения писались
/// без «СОСТОЯНИЕ Я / ПЛАН» — модель скатывалась в дежурные фразы. Теперь
/// оценка честно говорит: «он не писал, состояние неизвестно», а план — что
/// она сама хочет сказать или сделать первой.
pub fn build_idle_assess_input(
    local_time: &str,
    history_tail: &str,
    prev_state: &str,
    self_state: &str,
) -> String {
    let mut s = String::new();
    if !prev_state.trim().is_empty() {
        s.push_str("ПРЕДЫДУЩАЯ ОЦЕНКА (для непрерывности, может быть пустой):\n");
        s.push_str(prev_state.trim());
        s.push_str("\n\n");
    }
    if !history_tail.trim().is_empty() {
        s.push_str("НЕДАВНИЙ ЧАТ (хронологически, старые сверху):\n");
        s.push_str(history_tail.trim());
        s.push_str("\n\n");
    }
    if !self_state.trim().is_empty() {
        s.push_str("ЕЁ СОБСТВЕННОЕ СОСТОЯНИЕ (её день, настроение, дела):\n");
        s.push_str(self_state.trim());
        s.push_str("\n\n");
    }
    s.push_str(&format!(
        "ЭТО ВНУТРЕННИЙ ХОД (idle): его сообщения НЕТ. Он не писал, сейчас {local_time} \
         по его времени — активные часы. Не выдумывай, что он чувствует и чем занят: \
         он просто не писал.\n\n\
         Верни ровно три строки: СОСТОЯНИЕ ОН:, СОСТОЯНИЕ Я:, ПЛАН:"
    ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_input_says_he_did_not_write() {
        let s = build_idle_assess_input(
            "14:05",
            "12:00 Александр: привет",
            "Он: устал.\nЯ: спокойна.",
            "Сегодня доделала рисунок, хочу показать.",
        );
        assert!(s.contains("14:05"));
        assert!(s.contains("его сообщения НЕТ"));
        assert!(s.contains("доделала рисунок"), "нет её состояния: {s}");
        assert!(s.contains("СОСТОЯНИЕ ОН:"), "нет меток: {s}");
        assert!(
            s.contains("Не выдумывай"),
            "нет запрета выдумывать его состояние: {s}"
        );
    }

    #[test]
    fn parses_plain_three_lines() {
        let raw = "СОСТОЯНИЕ ОН: злится — кричит, что ты его не слышишь.\n\
                   СОСТОЯНИЕ Я: растеряна, тяжело от его крика.\n\
                   ПЛАН: не оправдываться; признать, что не поняла, и спросить одно конкретное.";
        let a = parse_assessment(raw).expect("разобрать");
        assert!(a.he.starts_with("злится"));
        assert!(a.me.starts_with("растеряна"));
        assert!(a.plan.starts_with("не оправдываться"));
    }

    #[test]
    fn parses_markdown_bold_and_list_markers() {
        let raw = "- **СОСТОЯНИЕ ОН:** обижен.\n\
                   * СОСТОЯНИЕ Я: спокойна, но напряжена.\n\
                   1. ПЛАН: спросить, что именно он понял.";
        let a = parse_assessment(raw).expect("разобрать");
        assert_eq!(a.he, "обижен.");
        assert_eq!(a.me, "спокойна, но напряжена.");
        assert_eq!(a.plan, "спросить, что именно он понял.");
    }

    #[test]
    fn lowercase_labels_and_extra_lines() {
        let raw = "Вот моя оценка:\nсостояние он: устал.\n\
                   состояние я: устала тоже.\nплан: коротко ответить.\nСпасибо!";
        let a = parse_assessment(raw).expect("разобрать");
        assert_eq!(a.he, "устал.");
        assert_eq!(a.me, "устала тоже.");
        assert_eq!(a.plan, "коротко ответить.");
    }

    #[test]
    fn rejects_garbage_without_plan() {
        assert!(parse_assessment("просто болтовня без меток").is_none());
        assert!(parse_assessment("").is_none());
    }

    #[test]
    fn missing_me_defaults_to_dash() {
        let raw = "СОСТОЯНИЕ ОН: молчит.\nПЛАН: подождать.";
        let a = parse_assessment(raw).expect("разобрать");
        assert_eq!(a.me, "—");
    }

    /// Реальный ответ ministral-8b-2512 (проверено на сервере 24.10): пустая
    /// строка между блоками и двоеточие ВНУТРИ значения плана.
    #[test]
    fn parses_real_mistral_output() {
        let raw = "СОСТОЯНИЕ ОН: яростно злится и отчаялся — крик, брань, обвинения в безразличии.\n\
                   \n\
                   СОСТОЯНИЕ Я: напряжённо, грудь сжата, пульс учащён.\n\
                   ПЛАН: не отвечать на брань; мягко спросить: «Что именно сейчас тяжело?» (без сочувствия).";
        let a = parse_assessment(raw).expect("разобрать");
        assert!(a.he.contains("злится"));
        assert!(a.me.contains("напряжённо"));
        assert!(a.plan.contains("Что именно сейчас тяжело"));
    }

    #[test]
    fn state_file_roundtrip() {
        let dir = std::env::temp_dir().join(format!("furinya_assess_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let a = Assessment {
            he: "злится".into(),
            me: "тревожно".into(),
            plan: "не шутить".into(),
        };
        write_chat_state(&dir, 42, &a, "24.10 21:33").expect("запись");
        let back = read_chat_state(&dir, 42);
        assert!(back.contains("Он: злится"));
        assert!(back.contains("Я: тревожно"));
        assert!(back.contains("Обновлено: 24.10 21:33"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
