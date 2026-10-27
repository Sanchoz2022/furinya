//! Ситуационная память: атомарная, последовательная история «что происходит
//! между нами прямо сейчас».
//!
//! Отличие от дневника (RAG-факты) и пер-чат выжимки (профиль+эпизоды):
//! это БЫСТРАЯ краткосрочная заметка на каждый обмен. После каждого хода
//! (живой/idle/сон) отдельным дешёвым LLM-вызовом дописывается ОДНА заметка:
//! ситуация словами + оценка ЕГО состояния + оценка МОЕГО состояния
//! (что, как, когда, после чего). История идёт атомарно и последовательно,
//! дописывается в конец — ничего не пережимается и не перемешивается.
//!
//! Хранилище (на чат, append-only):
//!   `data/diary/situation/<chat_id>.md`      — человекочитаемая история;
//!   `data/diary/situation/<chat_id>.jsonl`   — sidecar-индекс эмбеддингов
//!                                              (для семантического top-K).
//!
//! Чтение: последние N заметок (непрерывность) + семантический top-K по
//! свежему сообщению; плюс модель сама может достать конкретный фрагмент
//! тулом `situation` (№ / last / query).
//!
//! Файлы лежат в подпапке `situation/` — `Diary::load_all` читает только `.md`
//! в КОРНЕ diary_dir и не заглядывает сюда, так что с общим RAG не смешивается.

use std::path::PathBuf;

use anyhow::Result;
use furinya_core::prompts::Prompts;
use furinya_llm::{LlmClient, Message, MessageContent};
use tracing::{info, warn};

use crate::diary::{CENTER_MIN_ENTRIES, centered_cosine, corpus_mean};
use crate::embedding::EmbeddingClient;

/// Одна ситуационная заметка — атомарный блок одного обмена.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SituationNote {
    /// Порядковый номер (сквозная нумерация по чату, 1..).
    pub n: u32,
    /// Строка-заголовок, напр. `[23.10 21:14] №37`.
    pub header: String,
    /// Тело заметки: `Ситуация: …\nОн: …\nЯ: …`.
    pub body: String,
    /// Эмбеддинг (для семантического поиска). Может быть пустым, если
    /// сервис эмбеддингов недоступен — тогда заметка всё равно сохраняется.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embedding: Vec<f32>,
}

impl SituationNote {
    /// Полный блок в формате файла (`header\nbody`).
    pub fn block(&self) -> String {
        format!("{}\n{}", self.header, self.body)
    }
}

/// Хранилище ситуационной памяти ОДНОГО чата.
pub struct SituationStore {
    dir: PathBuf,
    chat_id: i64,
    path: PathBuf,
}

impl SituationStore {
    /// `diary_dir` — корень памяти (`data/diary`), `chat_id` — чат.
    pub fn new(diary_dir: &str, chat_id: i64) -> Self {
        let dir = std::path::Path::new(diary_dir).join("situation");
        let path = dir.join(format!("{}.md", chat_id));
        Self {
            dir,
            chat_id,
            path,
        }
    }

    fn index_path(&self) -> PathBuf {
        self.dir.join(format!("{}.jsonl", self.chat_id))
    }

    /// Разбирает markdown-историю в заметки (эмбеддинги — пустые; они в sidecar).
    fn parse_md(&self) -> Vec<SituationNote> {
        let content = std::fs::read_to_string(&self.path).unwrap_or_default();
        let mut notes: Vec<SituationNote> = Vec::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') && trimmed.contains('№') {
                let n = trimmed
                    .split('№')
                    .nth(1)
                    .map(|s| {
                        s.trim()
                            .chars()
                            .take_while(|c| c.is_ascii_digit())
                            .collect::<String>()
                    })
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0);
                notes.push(SituationNote {
                    n,
                    header: trimmed.to_string(),
                    body: String::new(),
                    embedding: Vec::new(),
                });
            } else if let Some(last) = notes.last_mut() {
                if !trimmed.is_empty() {
                    if !last.body.is_empty() {
                        last.body.push('\n');
                    }
                    last.body.push_str(trimmed);
                }
            }
        }
        notes
    }

    /// Загружает sidecar-индекс эмбеддингов из JSONL.
    fn load_index(&self) -> Vec<SituationNote> {
        let mut out = Vec::new();
        if let Ok(content) = std::fs::read_to_string(self.index_path()) {
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(note) = serde_json::from_str::<SituationNote>(line) {
                    out.push(note);
                }
            }
        }
        out
    }

    /// Сколько заметок уже есть (по истории на диске).
    pub fn count(&self) -> usize {
        self.parse_md().len()
    }

    /// Последние `n` заметок (непрерывность ситуации), склеенные в текст.
    pub fn recent(&self, n: usize) -> String {
        if n == 0 {
            return String::new();
        }
        let notes = self.parse_md();
        let start = notes.len().saturating_sub(n);
        notes[start..]
            .iter()
            .map(|x| x.block())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Конкретная заметка по номеру (для тула `situation`).
    pub fn block_by_number(&self, n: u32) -> Option<String> {
        self.parse_md()
            .into_iter()
            .find(|x| x.n == n)
            .map(|x| x.block())
    }

    /// Догоняет sidecar-индекс: для заметок без эмбеддинга считает его.
    /// Идемпотентно; ошибки эмбеддинга не фатальны (заметка остаётся без вектора).
    pub async fn ensure_index(&self, embedder: &EmbeddingClient) {
        let notes = self.parse_md();
        if notes.is_empty() {
            return;
        }
        let existing = self.load_index();
        let have: std::collections::HashSet<u32> = existing
            .iter()
            .filter(|n| !n.embedding.is_empty())
            .map(|n| n.n)
            .collect();
        let missing: Vec<&SituationNote> = notes
            .iter()
            .filter(|n| !have.contains(&n.n))
            .collect();
        if missing.is_empty() {
            return;
        }
        let texts: Vec<String> = missing.iter().map(|n| n.body.clone()).collect();
        let vecs = match embedder.embed_batch(&texts).await {
            Ok(v) => v,
            Err(e) => {
                warn!("[situation] индекс: эмбеддинги недоступны ({e})");
                return;
            }
        };
        // Собираем новый индекс: существующие векторы + свежие.
        let mut by_n: std::collections::HashMap<u32, Vec<f32>> = existing
            .into_iter()
            .map(|n| (n.n, n.embedding))
            .collect();
        for (note, v) in missing.iter().zip(vecs) {
            by_n.insert(note.n, v);
        }
        let rebuilt: Vec<String> = notes
            .iter()
            .filter_map(|n| {
                let embedding = by_n.remove(&n.n).unwrap_or_default();
                let full = SituationNote {
                    n: n.n,
                    header: n.header.clone(),
                    body: n.body.clone(),
                    embedding,
                };
                serde_json::to_string(&full).ok()
            })
            .collect();
        let _ = std::fs::create_dir_all(&self.dir);
        let _ = std::fs::write(self.index_path(), rebuilt.join("\n") + "\n");
        info!("[situation] индекс догнан: +{} заметок", missing.len());
    }

    /// Полный пересчёт бокового индекса под текущую модель: стирает старый
    /// jsonl и эмбеддит все заметки заново (пачками). Нужен при смене
    /// эмбеддера. Ошибка эмбеддинга пробрасывается — вызывающий не пишет
    /// маркер и повторит на следующем старте.
    pub async fn reindex(&self, embedder: &EmbeddingClient) -> anyhow::Result<usize> {
        let notes = self.parse_md();
        let _ = std::fs::create_dir_all(&self.dir);
        if notes.is_empty() {
            let _ = std::fs::write(self.index_path(), "");
            return Ok(0);
        }
        let mut lines: Vec<String> = Vec::new();
        let mut done = 0usize;
        // Терпеливый клиент: на 429 ждёт Retry-After провайдера, чтобы
        // переиндексация не падала из-за минутного лимита токенов.
        let embedder = embedder.patient();
        for (i, chunk) in notes.chunks(4).enumerate() {
            let texts: Vec<String> = chunk.iter().map(|n| n.body.clone()).collect();
            let vecs = embedder.embed_batch(&texts).await?;
            for (n, v) in chunk.iter().zip(vecs) {
                let full = SituationNote {
                    n: n.n,
                    header: n.header.clone(),
                    body: n.body.clone(),
                    embedding: v,
                };
                if let Ok(s) = serde_json::to_string(&full) {
                    lines.push(s);
                    done += 1;
                }
            }
            if (i + 1) * 4 < notes.len() {
                tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
            }
        }
        std::fs::write(self.index_path(), lines.join("\n") + "\n")?;
        Ok(done)
    }

    /// Семантический top-K по запросу. Пусто, если индекс ещё не построен.
    pub async fn search(
        &self,
        embedder: &EmbeddingClient,
        query: &str,
        k: usize,
    ) -> String {
        if k == 0 || query.trim().is_empty() {
            return String::new();
        }
        self.ensure_index(embedder).await;
        let notes = self.load_index();
        let scored: Vec<&SituationNote> = notes.iter().filter(|n| !n.embedding.is_empty()).collect();
        if scored.is_empty() {
            return String::new();
        }
        // Центрированный косинус: у mistral-embed сырое сходство заметок
        // почти не различается (общий «фон» ~0.9). Вычитаем средний вектор
        // корпуса. На маленьком индексе центрирование нестабильно — обычный.
        let mean = if scored.len() >= CENTER_MIN_ENTRIES {
            corpus_mean(scored.iter().map(|n| n.embedding.as_slice()))
        } else {
            Vec::new()
        };
        let qv = match embedder.embed(query).await {
            Ok(v) => v,
            Err(e) => {
                warn!("[situation] поиск: эмбеддинг запроса не удался ({e})");
                return String::new();
            }
        };
        let mut hits: Vec<(f32, &SituationNote)> = scored
            .into_iter()
            .map(|n| (centered_cosine(&qv, &n.embedding, &mean), n))
            .collect();
        hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k);
        hits.iter()
            .map(|(_, n)| n.block())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Дописывает ОДНУ заметку об этом обмене (LLM, temp ~0.3).
    ///
    /// `exchange` — новый диалог в формате `[ДД.ММ ЧЧ:ММ] Имя: текст`.
    /// `his_present` — было ли в обмене ЕГО сообщение. На idle/сон-ходе (она
    /// написала первой, он ещё не отвечал) его состояние объективно неизвестно,
    /// поэтому строку `Он:` принудительно заменяем честной — без выдумок.
    /// Возвращает сохранённую заметку (или None, если писать нечего/не вышло).
    pub async fn write_note(
        &self,
        llm: &LlmClient,
        prompts: &Prompts,
        embedder: &EmbeddingClient,
        exchange: &str,
        keep_entries: usize,
        his_present: bool,
    ) -> Result<Option<SituationNote>> {
        let system = prompts.situation();
        if system.trim().is_empty() || exchange.trim().len() < 8 {
            return Ok(None);
        }
        let notes = self.parse_md();
        let n = notes.iter().map(|x| x.n).max().unwrap_or(0) + 1;
        let now = chrono::Local::now();
        let header = format!("{} №{}", now.format("[%d.%m %H:%M]"), n);

        // Предыдущую заметку НЕ передаём. Модель принимала её за факты и
        // переносила в новую («он ответил спасибо на мою фантазию о…»), хотя
        // в обмене этого не было. Заметка атомарна: обмен — единственный вход.
        let input = format!(
            "НОВЫЙ ОБМЕН (единственный источник фактов; один чат, хронологически, \
             старые сверху, формат [ДД.ММ ЧЧ:ММ] Имя: текст):\n{}\n\n\
             Верни ТОЛЬКО заметку по формату (без нумерации и без заголовка с датой).",
            exchange
        );
        let resp = llm
            .chat(vec![Message::system(system), Message::user(input)], None)
            .await?;
        let raw = resp
            .choices
            .first()
            .and_then(|c| match &c.message.content {
                MessageContent::Text(s) => Some(s.clone()),
                MessageContent::Parts(_) => None,
            })
            .unwrap_or_default();
        let Some(mut body) = normalize_note(&raw) else {
            warn!("[situation] модель не вернула валидную заметку — пропускаю");
            return Ok(None);
        };
        // Idle/сон: его реплики не было — не даём модели выдумать его состояние.
        if !his_present {
            body = force_unknown_he(&body);
        }

        // Дописываем в markdown-историю (append-only, блоки через пустую строку).
        std::fs::create_dir_all(&self.dir)?;
        let mut md = std::fs::read_to_string(&self.path).unwrap_or_default();
        if !md.trim().is_empty() {
            if !md.ends_with("\n\n") {
                if !md.ends_with('\n') {
                    md.push('\n');
                }
                md.push('\n');
            }
        }
        md.push_str(&header);
        md.push('\n');
        md.push_str(&body);
        md.push('\n');
        std::fs::write(&self.path, &md)?;

        // Эмбеддинг заметки → sidecar-индекс (append-only jsonl).
        let embedding = embedder.embed(&body).await.unwrap_or_default();
        let note = SituationNote {
            n,
            header,
            body,
            embedding,
        };
        match serde_json::to_string(&note) {
            Ok(line) => {
                let mut idx = std::fs::read_to_string(self.index_path()).unwrap_or_default();
                if !idx.is_empty() && !idx.ends_with('\n') {
                    idx.push('\n');
                }
                idx.push_str(&line);
                idx.push('\n');
                let _ = std::fs::write(self.index_path(), idx);
            }
            Err(e) => warn!("[situation] индекс: сериализация не удалась ({e})"),
        }

        if keep_entries > 0 {
            self.rotate(keep_entries);
        }
        info!("[situation] заметка №{} добавлена", n);
        Ok(Some(note))
    }

    /// Ротация: держим на диске последние `keep_entries` заметок (md + jsonl).
    fn rotate(&self, keep_entries: usize) {
        let notes = self.parse_md();
        if notes.len() <= keep_entries {
            return;
        }
        let start = notes.len() - keep_entries;
        let kept = &notes[start..];
        let md: String = kept
            .iter()
            .map(|x| x.block())
            .collect::<Vec<_>>()
            .join("\n\n")
            + "\n";
        let _ = std::fs::write(&self.path, md);
        let kept_n: std::collections::HashSet<u32> = kept.iter().map(|x| x.n).collect();
        let idx: Vec<String> = self
            .load_index()
            .into_iter()
            .filter(|n| kept_n.contains(&n.n))
            .filter_map(|n| serde_json::to_string(&n).ok())
            .collect();
        let _ = std::fs::write(self.index_path(), idx.join("\n") + "\n");
        info!(
            "[situation] ротация: оставлено {} последних заметок",
            keep_entries
        );
    }
}

/// Нормализует сырой ответ LLM в тело заметки строго из трёх строк.
///
/// Устойчиво к markdown-обёрткам (`**`, `- `, код-фенсы, преамбулы):
/// берём только строки с ключами `Ситуация:`/`Он:`/`Я:`. Без `Ситуация:`
/// заметка невалидна; `Он:`/`Я:` хотя бы одно должны быть.
fn normalize_note(raw: &str) -> Option<String> {
    let mut sit: Option<String> = None;
    let mut he: Option<String> = None;
    let mut me: Option<String> = None;
    for line in raw.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("```") {
            continue;
        }
        let t = t.trim_start_matches(['-', '*', '#']).trim();
        let t = t.trim_matches('*').trim();
        let Some((k, v)) = t.split_once(':') else {
            continue;
        };
        let key = k.trim().trim_matches('*').trim().to_lowercase();
        // Срезаем markdown-обёртку и у значения (`** оживился` → `оживился`).
        let val = one_line(v.trim_matches(|c: char| c == '*' || c == '`' || c == ' '));
        if val.is_empty() {
            continue;
        }
        match key.as_str() {
            "ситуация" | "situation" => sit = Some(val),
            "он" | "его состояние" | "состояние саши" | "he" => he = Some(val),
            "я" | "моё состояние" | "мое состояние" | "моё состояние с ним" | "i" | "she" => {
                me = Some(val)
            }
            _ => {}
        }
    }
    let sit = sit?;
    if he.is_none() && me.is_none() {
        return None;
    }
    Some(format!(
        "Ситуация: {}\nОн: {}\nЯ: {}",
        sit,
        he.unwrap_or_else(|| "—".to_string()),
        me.unwrap_or_else(|| "—".to_string())
    ))
}

/// Схлопывает значение в одну строку (без переводов) и ограничивает длину.
/// При превышении лимита режет по последнему концу предложения (чтобы не
/// рвать мысль посреди слова) — модель склонна раздувать «Ситуация:» до
/// литературного абзаца, а заметка должна оставаться атомарной.
fn one_line(s: &str) -> String {
    let collapsed: String = s
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    let squashed = collapsed.split_whitespace().collect::<Vec<_>>().join(" ");
    const LIMIT: usize = 230;
    if squashed.chars().count() <= LIMIT {
        return squashed;
    }
    let head: String = squashed.chars().take(LIMIT).collect();
    let mut cut: Option<usize> = None;
    for (i, c) in head.char_indices() {
        if c == '.' || c == '!' || c == '?' {
            cut = Some(i + c.len_utf8());
        }
    }
    if let Some(end) = cut {
        if end >= 80 {
            return head[..end].trim_end().to_string();
        }
    }
    head.trim_end().to_string()
}

/// Заменяет строку `Он:` честным «его не было» — для idle/сон-ходов, где он
/// ещё не отвечал и модель склонна выдумать его состояние.
fn force_unknown_he(body: &str) -> String {
    let mut out = String::new();
    for line in body.lines() {
        if line.starts_with("Он:") {
            out.push_str("Он: он сейчас не писал — его состояние неизвестно.");
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_note_parses_canonical() {
        let raw = "Ситуация: он прислал фото гитары.\nОн: гордится.\nЯ: тепло и радость.";
        let note = normalize_note(raw).unwrap();
        assert!(note.starts_with("Ситуация: он прислал фото гитары."));
        assert!(note.contains("Он: гордится."));
        assert!(note.contains("Я: тепло и радость."));
    }

    #[test]
    fn normalize_note_tolerates_markdown_and_preamble() {
        let raw = "Вот заметка:\n\n**Ситуация:** обсуждали музыку\n- **Он:** оживился\n- **Я:** вдохновилась";
        let note = normalize_note(raw).unwrap();
        assert_eq!(
            note,
            "Ситуация: обсуждали музыку\nОн: оживился\nЯ: вдохновилась"
        );
    }

    #[test]
    fn normalize_note_rejects_garbage() {
        assert!(normalize_note("просто болтовня без структуры").is_none());
        assert!(normalize_note("Ситуация: есть, но без оценок вообще").is_none());
    }

    #[test]
    fn force_unknown_he_replaces_his_line() {
        let body = "Ситуация: написала первой.\nОн: возможно, застыл над экраном.\nЯ: тепло.";
        let out = force_unknown_he(body);
        assert!(out.contains("Он: он сейчас не писал — его состояние неизвестно."));
        assert!(!out.contains("застыл"));
        assert!(out.contains("Ситуация: написала первой."));
        assert!(out.contains("Я: тепло."));
    }

    #[test]
    fn one_line_caps_at_sentence_boundary() {
        let long = format!(
            "{}конец. {}",
            "слово ".repeat(33),
            "и ещё очень длинный хвост который точно перевалит за лимит"
        );
        let out = one_line(&long);
        assert!(out.chars().count() <= 230);
        assert!(out.ends_with("конец."), "got: {out}");
    }

    #[test]
    fn one_line_keeps_short_text() {
        assert_eq!(one_line("  коротко\nи  ясно "), "коротко и ясно");
    }

    #[test]
    fn store_parses_blocks_and_recent() {
        let dir = std::env::temp_dir().join(format!("furinya_sit_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("situation")).unwrap();
        let path = dir.join("situation").join("1.md");
        std::fs::write(
            &path,
            "[23.10 21:14] №1\nСитуация: первая.\nОн: спокоен.\nЯ: ровно.\n\n\
             [23.10 21:40] №2\nСитуация: вторая.\nОн: устал.\nЯ: заботливо.\n",
        )
        .unwrap();
        let store = SituationStore::new(dir.to_str().unwrap(), 1);
        assert_eq!(store.count(), 2);
        let recent = store.recent(1);
        assert!(recent.contains("№2"));
        assert!(recent.contains("Ситуация: вторая."));
        assert!(!recent.contains("№1"));
        assert!(store.block_by_number(1).unwrap().contains("Ситуация: первая."));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
