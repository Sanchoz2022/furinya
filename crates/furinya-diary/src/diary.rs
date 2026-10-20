//! Дневник (постоянная векторная память / RAG).
//!
//! Каждая запись хранится в отдельном `.md` файле: в начале — JSON-метаданные
//! в блоке `---`, затем текст воспоминания (с тегами `#Тема`, служебными
//! строками `- thought:` / `- emotion:` и т.д.).
//!
//! Поведение взято из kuni (Diary.cpp / diary_save_entries.h):
//! - сохранение: split по `---`, эмбеддинг каждой записи, plagiarism-check
//!   (relatedness > threshold → дубликат, не сохраняем), запись на диск;
//! - поиск: косинусное сходство эмбеддинга запроса с записями,
//!   фильтр по `min_relatedness`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use serde::Serialize;

use crate::active::{active_context, corrupt_set, read_active, write_active};
use crate::embedding::EmbeddingClient;

/// Метаданные записи дневника (сериализуются в JSON в блоке `---`).
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct EntryMetadata {
    #[serde(default)]
    pub score: f32,
    #[serde(default)]
    pub confidence: f32,
    #[serde(default = "default_last_used")]
    pub last_used: String,
    #[serde(default)]
    pub usage_count: u32,
    #[serde(default)]
    pub embedding: Vec<f32>,
    /// Теги (`#Музыка`, `#Александр`), извлечённые из body моделью.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Помечена ли запись как «битая» (неудачная консолидация). Такие записи
    /// исключаются из активного контекста и RAG, но хранятся для отката.
    #[serde(default)]
    pub corrupt: bool,
}

fn default_last_used() -> String {
    "never".to_string()
}

/// Запись дневника: идентификатор (имя файла без .md) + метаданные + тело.
#[derive(Debug, Clone)]
pub struct Entry {
    pub id: String,
    pub metadata: EntryMetadata,
    pub body: String,
}

/// Результат поиска: запись + нормированное сходство [0..1].
#[derive(Debug, Clone)]
pub struct QueryHit {
    pub entry: Entry,
    pub relatedness: f32,
}

/// Косинусное сходство двух векторов.
/// Возвращает 0.0 при нулевой длине (как в kuni cosine_similarity.h).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Дневник на диске (каталог `data/diary/*.md`).
pub struct Diary {
    dir: PathBuf,
    embedding: EmbeddingClient,
    /// Порог косинуса для отсечения дубликатов при сохранении.
    plagiarism_threshold: f32,
    /// Минимальный порог сходства для релевантности при поиске.
    min_relatedness: f32,
}

impl Diary {
    pub fn new(
        dir: impl AsRef<Path>,
        embedding: EmbeddingClient,
        plagiarism_threshold: f32,
        min_relatedness: f32,
    ) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            embedding,
            plagiarism_threshold,
            min_relatedness,
        }
    }

    /// Загружает все записи из каталога.
    pub fn load_all(&self) -> anyhow::Result<Vec<Entry>> {
        std::fs::create_dir_all(&self.dir)
            .with_context(|| format!("создание каталога дневника {:?}", self.dir))?;
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            let text = std::fs::read_to_string(&path)?;
            entries.push(Self::parse_file(&path, &text));
        }
        Ok(entries)
    }

    /// Парсит один `.md` файл: выкусывает блок `---` с JSON, остальное — тело.
    fn parse_file(path: &Path, text: &str) -> Entry {
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        if let Some((meta_str, body)) = split_meta(text) {
            match serde_json::from_str::<EntryMetadata>(meta_str) {
                Ok(mut metadata) => {
                    // Теги: берём из метаданных, если есть, иначе парсим из body.
                    if metadata.tags.is_empty() {
                        metadata.tags = extract_tags(body);
                    }
                    Entry {
                        id,
                        metadata,
                        body: body.trim().to_string(),
                    }
                }
                Err(_) => Entry {
                    id,
                    metadata: EntryMetadata::default(),
                    body: text.trim().to_string(),
                },
            }
        } else {
            Entry {
                id,
                metadata: EntryMetadata::default(),
                body: text.trim().to_string(),
            }
        }
    }

    /// Сохраняет запись на диск (`<id>.md`).
    pub fn save(&self, entry: &Entry) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let meta_json = serde_json::to_string(&entry.metadata)?;
        let content = format!("---\n{}\n---\n{}", meta_json, entry.body);
        let path = self.dir.join(format!("{}.md", entry.id));
        std::fs::write(&path, content).with_context(|| format!("запись {}", path.display()))?;
        Ok(())
    }

    /// Сохраняет запись на диск (`<id>.md`) и сразу выполняет жёсткую ротацию
    /// (prune_old): если итоговых записей больше `keep_last`, самые старые по
    /// timestamp удаляются. Защищены от удаления: активный good-контекст
    /// (топ-K по `active_top_k` из active.txt) + все corrupt-записи (нужны для
    /// rollback). Это и есть «цепочка»: создали новую запись → вышедшие за
    /// лимит старые удаляются, active всегда живой.
    ///
    /// `entries` — текущее состояние в памяти (ДО записи этой записи). Метод
    /// сам дочитывает свеженаписанный файл и вызывает prune.
    pub fn save_entry(
        &self,
        entry: &Entry,
        entries: &[Entry],
        keep_last: usize,
        active_top_k: usize,
    ) -> anyhow::Result<usize> {
        self.save(entry)?;
        if keep_last == 0 {
            return Ok(0);
        }
        // Перезагружаем, чтобы prune видел и только что записанный файл.
        let mut all = self.load_all().unwrap_or_else(|_| entries.to_vec());
        if all.iter().all(|e| e.id != entry.id) {
            all.push(entry.clone());
        }
        // Защита: активный контекст из active.txt + corrupt-записи на диске.
        let (active_ts_str, corrupt) = read_active(&self.dir);
        let mut protected: HashSet<String> = corrupt_set(&corrupt);
        // active_ts один, но активный КОНТЕКСТ — это топ-K вокруг него.
        // Берём топ-K самых свежих good записей (как в consolidate.rs).
        let active_ctx = active_context(&all, active_top_k, &corrupt);
        for e in &active_ctx {
            protected.insert(e.id.clone());
        }
        // Если active.txt пуст ИЛИ ссылается на осиротевший id (файл удалён /
        // перемещён в _trash при чистке мусора) — защищаем саму свеженаписанную
        // запись, чтобы не удалить то, что только создали, и не держать
        // мёртвую ссылку в protected.
        let active_orphan = !active_ts_str.is_empty()
            && all.iter().all(|e| e.id != active_ts_str);
        if active_ts_str.is_empty() || active_orphan {
            protected.insert(entry.id.clone());
            // Чиним осиротевший/пустой active.txt: переписываем на свежую запись,
            // чтобы активный указатель бота больше не ссылался на удалённый файл.
            let _ = write_active(&self.dir, &entry.id, &corrupt);
        }
        let removed = self.prune_old(&all, keep_last, &protected)?;
        Ok(removed)
    }

    /// Публичный доступ к эмбеддинг-клиенту (для исполнения #ask).
    pub fn embedding(&self) -> &EmbeddingClient {
        &self.embedding
    }

    /// Возвращает отсортированные по relatedness (по убыванию) хиты выше порога.
    pub fn query(&self, entries: &[Entry], query_embedding: &[f32]) -> Vec<QueryHit> {
        let mut hits: Vec<QueryHit> = entries
            .iter()
            // Битые записи (metadata.corrupt) в выдачу НЕ идут: они помечены
            // после неудачной консолидации и могут содержать мусор.
            .filter(|e| !e.metadata.corrupt)
            .filter(|e| !e.metadata.embedding.is_empty())
            .map(|e| QueryHit {
                entry: e.clone(),
                relatedness: cosine_similarity(query_embedding, &e.metadata.embedding),
            })
            .filter(|h| h.relatedness >= self.min_relatedness)
            .collect();
        hits.sort_by(|a, b| {
            b.relatedness
                .partial_cmp(&a.relatedness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits
    }

    /// То же, что `query`, но ограничивает выдачу `n` самыми релевантными
    /// записями. Нужно, чтобы модель при `#ask` получала КОНТЕКСТНЫЕ отрывки,
    /// а не весь дневник целиком (иначе раздувается контекст LLM и падает
    /// exceed_context_size).
    pub fn query_top_n(
        &self,
        entries: &[Entry],
        query_embedding: &[f32],
        n: usize,
    ) -> Vec<QueryHit> {
        let mut hits = self.query(entries, query_embedding);
        hits.truncate(n);
        hits
    }

    /// Сохраняет записи из текста LLM (разбитого по `---`).
    ///
    /// Для каждой части > `min_len` символов:
    /// 1. эмбеддинг;
    /// 2. plagiarism-check против существующих (relatedness > plagiarism_threshold → скип);
    /// 3. запись на диск с метаданными.
    ///
    /// Возвращает сохранённые записи.
    pub async fn save_entries_from_text(
        &self,
        text: &str,
        existing: &[Entry],
        min_len: usize,
        keep_last: usize,
        active_top_k: usize,
    ) -> anyhow::Result<Vec<Entry>> {
        let mut saved = Vec::new();
        let ts_base = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut ts = ts_base;

        for part in text
            .split("\n---\n")
            .map(|s| s.trim())
            .filter(|s| s.len() >= min_len)
        {
            // Защита от мусора: скипаем "memory map empty / карта памяти пуста"
            // и прочие мета-фразы о пустом состоянии. Они не несут фактов и
            // только засоряют RAG (проверено: 56 таких записей накопилось).
            if is_empty_state_noise(part) {
                eprintln!("[diary] пропуск мусорной записи 'empty state'");
                continue;
            }

            let embedding = self.embedding.embed(part).await?;

            // plagiarism check
            let dup = existing
                .iter()
                .filter(|e| !e.metadata.embedding.is_empty())
                .map(|e| cosine_similarity(&embedding, &e.metadata.embedding))
                .any(|sim| sim > self.plagiarism_threshold);
            if dup {
                eprintln!(
                    "[diary] пропуск дубликата (plagiarism > {})",
                    self.plagiarism_threshold
                );
                continue;
            }

            let entry = Entry {
                id: ts.to_string(),
                metadata: EntryMetadata {
                    score: 0.0,
                    confidence: 0.0,
                    last_used: now_string(),
                    usage_count: 0,
                    embedding,
                    tags: extract_tags(part),
                    corrupt: false,
                },
                body: part.to_string(),
            };
            // Сохраняем + сразу режем старые (цепочка: создал → за лимитом удалён).
            let removed = self.save_entry(&entry, existing, keep_last, active_top_k)?;
            if removed > 0 {
                tracing::info!("[diary] prune после записи: удалено {}", removed);
            }
            saved.push(entry);
            ts += 1; // гарантируем уникальный id в рамках одного вызова
        }
        Ok(saved)
    }
}

/// Делит текст на (json_meta, body) по первому блоку `--- ... ---`.
fn split_meta(text: &str) -> Option<(&str, &str)> {
    let trimmed = text.trim_start();
    if !trimmed.starts_with("---\n") && !trimmed.starts_with("---\r\n") {
        return None;
    }
    let after_open = &trimmed[4..]; // пропускаем "---\n" или "---\r\n" (4 символа в обоих случаях)
    let close = after_open.find("\n---")?;
    let meta = &after_open[..close];
    let body = &after_open[close + 4..];
    Some((meta, body))
}

impl Diary {
    /// Возвращает записи с confidence < 1 (изменяемые, подлежат консолидации).
    pub fn mutables<'a>(&self, entries: &'a [Entry]) -> Vec<&'a Entry> {
        entries
            .iter()
            .filter(|e| e.metadata.confidence < 1.0)
            .collect()
    }

    /// Удаляет «очень старые» записи дневника, оставляя последние `keep_last`
    /// по таймстампу (имя файла = unix-секунды). Защищённые записи НЕ
    /// удаляются никогда: `protected` — это множество id (обычно активный
    /// good-контекст топ-K + corrupt-записи, нужные для rollback).
    ///
    /// Возвращает количество реально удалённых файлов.
    pub fn prune_old(&self, entries: &[Entry], keep_last: usize, protected: &std::collections::HashSet<String>) -> anyhow::Result<usize> {
        if keep_last == 0 {
            return Ok(0);
        }
        // Сортируем по ts (id) по возрастанию → берём последние keep_last как «свежие».
        let mut sorted: Vec<&Entry> = entries.iter().collect();
        sorted.sort_by(|a, b| a.id.cmp(&b.id)); // id — это unix ts как строка, лексикографически == числовому для фикс. длины
        let fresh: std::collections::HashSet<String> = sorted
            .into_iter()
            .rev()
            .take(keep_last)
            .map(|e| e.id.clone())
            .collect();

        let mut removed = 0usize;
        for e in entries {
            if fresh.contains(&e.id) || protected.contains(&e.id) {
                continue;
            }
            let path = self.dir.join(format!("{}.md", e.id));
            if path.exists() {
                std::fs::remove_file(&path)
                    .with_context(|| format!("удаление старой записи {}", path.display()))?;
                removed += 1;
            }
        }
        Ok(removed)
    }

    /// Помечает запись `<id>.md` как битую (corrupt), перезаписывая её
    /// метаданные (источник правды — флаг в --- JSON записи). Используется
    /// при неудачной консолидации для rollback-логики.
    pub fn mark_corrupt(&self, id: &str) -> anyhow::Result<()> {
        let path = self.dir.join(format!("{}.md", id));
        if !path.exists() {
            return Ok(());
        }
        let content = std::fs::read_to_string(&path)?;
        let (meta_str, body) = split_meta(&content)
            .with_context(|| format!("parse entry {} для пометки corrupt", id))?;
        let mut meta: EntryMetadata = serde_json::from_str(meta_str)
            .with_context(|| format!("parse metadata entry {}", id))?;
        meta.corrupt = true;
        let out = format!(
            "---\n{}\n---\n\n{}",
            serde_json::to_string_pretty(&meta).unwrap_or_default(),
            body
        );
        std::fs::write(&path, out)
            .with_context(|| format!("запись corrupt-метки в {}", path.display()))?;
        Ok(())
    }

    /// Возвращает записи, содержащие хотя бы один из указанных тегов.
    /// Если `tags` пуст — возвращает все записи (без фильтра).
    pub fn filter_by_tags<'a>(&self, entries: &'a [Entry], tags: &[String]) -> Vec<&'a Entry> {
        if tags.is_empty() {
            return entries.iter().collect();
        }
        entries
            .iter()
            .filter(|e| e.metadata.tags.iter().any(|t| tags.contains(t)))
            .collect()
    }

    /// Сохраняет результат консолидации: список (confidence, tags, body).
    /// Каждый кусок перезаписывается как отдельная запись (новый id по времени).
    /// После каждой записи — жёсткая ротация (save_entry → prune_old), так что
    /// итоговых файлов не больше `keep_last` (кроме защищённого active-контекста).
    pub async fn save_consolidated(
        &self,
        pieces: &[(f32, Vec<String>, String)],
        existing: &[Entry],
        keep_last: usize,
        active_top_k: usize,
    ) -> anyhow::Result<Vec<Entry>> {
        let mut saved = Vec::new();
        let ts_base = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut ts = ts_base;
        for (confidence, tags, body) in pieces {
            let embedding = self.embedding.embed(body).await?;
            // plagiarism check против существующих
            let dup = existing
                .iter()
                .filter(|e| !e.metadata.embedding.is_empty())
                .map(|e| cosine_similarity(&embedding, &e.metadata.embedding))
                .any(|sim| sim > self.plagiarism_threshold);
            if dup {
                continue;
            }
            let entry = Entry {
                id: ts.to_string(),
                metadata: EntryMetadata {
                    score: 0.0,
                    confidence: *confidence,
                    last_used: now_string(),
                    usage_count: 0,
                    embedding,
                    tags: tags.clone(),
                    corrupt: false,
                },
                body: body.clone(),
            };
            self.save_entry(&entry, existing, keep_last, active_top_k)?;
            saved.push(entry);
            ts += 1;
        }
        Ok(saved)
    }
}

/// Детект мусорных записей про «пустое состояние памяти» (memory map empty,
/// карта памяти пуста, no memories recorded и т.п.). Такие куски не несут фактов
/// и только засоряют RAG. Проверяем по набору ключевых фраз (англ+рус) в нижнем
/// регистре — этого достаточно, т.к. полезные факты содержат имена/события.
fn is_empty_state_noise(text: &str) -> bool {
    let t = text.to_lowercase();
    const PATTERNS: &[&str] = &[
        "memory map",
        "карта памяти",
        "memory is currently empty",
        "map is currently empty",
        "no specific memories",
        "no memories have been recorded",
        "no memories recorded",
        "empty state",
        "currently empty",
        "пустая карта",
        "память пуста",
        "пусто, карта",
        "карта пустая",
        "nothing has been recorded",
        "no real new information",
    ];
    // Пропускаем, если это КОРОТКАЯ мета-фраза (до ~400 симв) и содержит хотя бы
    // один паттерн. Длинные куски с реальными фактами (где случайно упомянута
    // "карта памяти") не трогаем — они полезны.
    if t.len() > 600 {
        return false;
    }
    PATTERNS.iter().any(|p| t.contains(p))
}

/// Извлекает хэштеги (`#Музыка`, `#Александр`) из текста записи.
/// Тег — `#` за которым идут буквы/цифры/`_`, до пробела или пунктуации.
/// Возвращает уникальный список (без `#`).
fn extract_tags(text: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '#' {
            let mut j = i + 1;
            let mut tag = String::new();
            while j < chars.len() {
                let c = chars[j];
                if c.is_alphanumeric() || c == '_' {
                    tag.push(c);
                    j += 1;
                } else {
                    break;
                }
            }
            if !tag.is_empty() && !tags.contains(&tag) {
                tags.push(tag);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    tags
}

/// Парсит ответ Sleep-time Consolidator (формат kuni):
/// последовательность кусков `{"confidence":X}\n\nbody`, разделённых `\n\n---\n\n`.
/// Возвращает (confidence, tags, body) для каждого куска.
pub fn parse_consolidated(text: &str) -> Vec<(f32, Vec<String>, String)> {
    let mut out = Vec::new();
    for chunk in text.split("\n\n---\n\n") {
        let chunk = chunk.trim();
        if chunk.is_empty() {
            continue;
        }
        let confidence = extract_confidence(chunk).unwrap_or(0.0);
        let tags = extract_tags(chunk);
        let body = strip_leading_json(chunk).trim().to_string();
        if !body.is_empty() {
            out.push((confidence, tags, body));
        }
    }
    out
}

/// Извлекает `confidence` из JSON-объекта в начале текста.
fn extract_confidence(text: &str) -> Option<f32> {
    let start = text.find('{')?;
    let end = text[start..].find('}')? + start + 1;
    let json_str = &text[start..end];
    serde_json::from_str::<serde_json::Value>(json_str)
        .ok()
        .and_then(|v| v.get("confidence")?.as_f64())
        .map(|f| f as f32)
}

/// Убирает ведущий JSON-объект (вплоть до первой пустой строки после него).
fn strip_leading_json(text: &str) -> String {
    if let Some(pos) = text.find('{') {
        let after = &text[pos..];
        if let Some(end) = after.find('}') {
            let rest = &after[end + 1..];
            return rest.trim_start_matches(['\n', '\r', ' ']).to_string();
        }
    }
    text.to_string()
}

fn now_string() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // Простой ISO-подобный вид; точное форматирование не критично для RAG.
    format!("{}", now)
}

impl Default for EntryMetadata {
    fn default() -> Self {
        Self {
            score: 0.0,
            confidence: 0.0,
            last_used: default_last_used(),
            usage_count: 0,
            embedding: Vec::new(),
            tags: Vec::new(),
            corrupt: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use furinya_core::EndpointConfig;

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn cosine_empty_is_zero() {
        let a: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &a), 0.0);
    }

    #[test]
    fn query_filters_by_threshold() {
        let entries = vec![
            Entry {
                id: "1".into(),
                metadata: EntryMetadata {
                    embedding: vec![1.0, 0.0, 0.0],
                    ..Default::default()
                },
                body: "alpha".into(),
            },
            Entry {
                id: "2".into(),
                metadata: EntryMetadata {
                    embedding: vec![0.0, 1.0, 0.0],
                    ..Default::default()
                },
                body: "beta".into(),
            },
        ];
        let diary = Diary::new(
            "data/diary_test_tmp",
            EmbeddingClient::from_config(&EndpointConfig::default()),
            0.97,
            0.35,
        );
        // query близок к "alpha"
        let hits = diary.query(&entries, &[0.9, 0.1, 0.0]);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].entry.id, "1");
    }
}
