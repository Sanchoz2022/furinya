//! Загрузка и сборка промптов ядра (миграция с kuni/src/prompts.cpp).
//!
//! Промпты лежат в папке `prompts/` рядом с конфигом. Внутри — markdown с
//! YAML front-matter (`---\n...\n---`), который вырезается. Поддерживаются
//! переменные `${CHARACTER_NAME}`, `${CHARACTER_NICKNAME}`, `${PAPIK_NAME}`.

use crate::Config;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PromptError {
    #[error("не удалось прочитать промпт {0}: {1}")]
    Io(PathBuf, #[source] std::io::Error),
}

/// Набор загруженных промптов.
#[derive(Debug, Clone, Default)]
pub struct Prompts {
    pub system: String,
    pub character_base: String,
    pub character_appearance: String,
    /// Прочие промпты по имени файла (без .md), напр. "diary_save".
    pub extra: HashMap<String, String>,
}

impl Prompts {
    /// Загружает промпты из папки `dir`, подставляя переменные из конфига.
    pub fn load_from_dir<P: AsRef<Path>>(dir: P, cfg: &Config) -> Result<Self, PromptError> {
        let dir = dir.as_ref();
        let vars = build_vars(cfg);

        let load = |name: &str| -> Result<String, PromptError> {
            let path = dir.join(format!("{name}.md"));
            let raw =
                std::fs::read_to_string(&path).map_err(|e| PromptError::Io(path.clone(), e))?;
            Ok(apply_vars(&strip_front_matter(&raw), &vars))
        };

        let system = load("system")?;
        let character_base = load("character_base")?;
        let character_appearance = load("character_appearance")?;

        // Доп. промпты (те, что не вошли в базовый набор).
        let mut extra = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("md") {
                    continue;
                }
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default()
                    .to_string();
                if matches!(
                    stem.as_str(),
                    "system" | "character_base" | "character_appearance"
                ) {
                    continue;
                }
                let raw =
                    std::fs::read_to_string(&path).map_err(|e| PromptError::Io(path.clone(), e))?;
                extra.insert(stem, apply_vars(&strip_front_matter(&raw), &vars));
            }
        }

        Ok(Self {
            system,
            character_base,
            character_appearance,
            extra,
        })
    }

    /// Собирает system-промпт для LLM: system + character_base
    /// + character_appearance + messages_epilogue (анти-инъекция, всегда)
    /// + anti_repeat (защита от повторов).
    pub fn build_system_prompt(&self) -> String {
        let mut s = String::new();
        s.push_str(&self.system);
        s.push_str("\n\n");
        s.push_str(&self.character_base);
        s.push_str("\n\n");
        s.push_str(&self.character_appearance);
        s.push_str("\n\n");
        s.push_str(&self.epilogue());
        s.push_str("\n\n");
        #[allow(clippy::collapsible_if)]
        if let Some(ar) = self.extra.get("anti_repeat") {
            if !ar.trim().is_empty() {
                s.push_str(ar);
                s.push_str("\n\n");
            }
        }
        s
    }

    /// Короткий защитный суффикс (messages_epilogue.md), который по
    /// задумке kuni вставляется ПРИ КАЖДОМ показе сообщений модели
    /// (анти-инъекция, "не будь ассистентом", влияние физ/эмоц состояния).
    pub fn epilogue(&self) -> String {
        self.extra
            .get("messages_epilogue")
            .cloned()
            .unwrap_or_default()
    }

    /// Только личность (character_base + appearance), БЕЗ workflow из system.md.
    /// Используется в терминальном/консольном режиме, где tool-call loop недоступен.
    pub fn build_character_prompt(&self) -> String {
        let mut s = String::new();
        s.push_str(&self.character_base);
        s.push_str("\n\n");
        s.push_str(&self.character_appearance);
        s
    }

    /// Промпт Sleep-time Consolidator (как в kuni: sleep_consolidator.md).
    /// Используется для консолидации/сжатия памяти.
    pub fn sleep_consolidator(&self) -> String {
        self.extra
            .get("sleep_consolidator")
            .cloned()
            .unwrap_or_default()
    }

    pub fn chat_history_consolidator(&self) -> String {
        self.extra
            .get("chat_history_consolidator")
            .cloned()
            .unwrap_or_default()
    }

    pub fn user_consolidator(&self) -> String {
        self.extra
            .get("user_consolidator")
            .cloned()
            .unwrap_or_default()
    }

    pub fn diary_save(&self) -> String {
        self.extra
            .get("diary_save")
            .cloned()
            .unwrap_or_default()
    }

    /// Промпт обновления «своего состояния» (inner state) — отдельный файл
    /// self_state.md, который держит её собственное настроение/занятия.
    pub fn self_state(&self) -> String {
        self.extra
            .get("self_state")
            .cloned()
            .unwrap_or_default()
    }

    /// Промпт ситуационной заметки (situation.md) — атомарная последовательная
    /// история «что происходит между нами прямо сейчас».
    pub fn situation(&self) -> String {
        self.extra
            .get("situation")
            .cloned()
            .unwrap_or_default()
    }

    /// Промпт внутренней оценки хода (assess.md) — «размышление» перед ответом:
    /// состояние его, своё состояние и план реплики.
    pub fn assess(&self) -> String {
        self.extra.get("assess").cloned().unwrap_or_default()
    }
}

/// Вырезает YAML front-matter в начале файла (блок между первым и вторым `---`).
fn strip_front_matter(raw: &str) -> String {
    let trimmed = raw.trim_start();
    if !trimmed.starts_with("---") {
        return raw.to_string();
    }
    // индекс конца первой строки `---`
    if let Some(after_first) = trimmed.find('\n') {
        let rest = &trimmed[after_first + 1..];
        // ищем закрывающий `\n---`
        if let Some(end) = rest.find("\n---") {
            return rest[end + 4..].trim_start().to_string();
        }
    }
    raw.to_string()
}

/// Строит карту переменных из конфига.
fn build_vars(cfg: &Config) -> HashMap<&'static str, String> {
    let mut m = HashMap::new();
    m.insert("CHARACTER_NAME", cfg.general.character_name.clone());
    m.insert("CHARACTER_NICKNAME", cfg.general.character_nickname.clone());
    m.insert("PAPIK_NAME", cfg.general.papik_name.clone());
    m
}

/// Подставляет ${VAR} из карты.
fn apply_vars(text: &str, vars: &HashMap<&'static str, String>) -> String {
    let mut out = text.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("${{{k}}}"), v);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_front_matter_basic() {
        let raw = "---\ntitle: x\n---\nHello world";
        assert_eq!(strip_front_matter(raw), "Hello world");
    }

    #[test]
    fn strip_front_matter_no_fm() {
        let raw = "just text";
        assert_eq!(strip_front_matter(raw), "just text");
    }

    #[test]
    fn apply_vars_works() {
        let mut vars = HashMap::new();
        vars.insert("CHARACTER_NAME", "Furinya".to_string());
        assert_eq!(apply_vars("hi ${CHARACTER_NAME}!", &vars), "hi Furinya!");
    }
}
