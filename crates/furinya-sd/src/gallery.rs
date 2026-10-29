//! Pre-generated photo gallery — кэш готовых фото для мгновенной отправки.
//!
//! **Проблема**: SD на CPU = 3.5 минуты ожидания, модель простаивает.
//! **Решение**: генерация пачками в фоне → сохранение годных (≥7/10) → быстрая выдача.
//!
//! **Архитектура**:
//! - `data/gallery/sfw/` — повседневные фото (улыбка, селфи, casual)
//! - `data/gallery/nsfw/` — откровенные фото (intimate)
//! - `data/gallery/metadata.json` — [{id, path, prompt, score, category, ts}, ...]
//!
//! **Workflow**:
//! 1. Background worker (tokio::spawn) генерит пачки каждые 30-60 мин (если кэш < min)
//! 2. Каждое фото проходит vision-оценку (Mistral Pixtral)
//! 3. Только ≥7/10 попадают в gallery
//! 4. При запросе `generate_image`: сначала поиск в gallery (мгновенно), fallback → live SD

use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;

/// Метаданные одного фото в gallery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalleryPhoto {
    /// Уникальный ID (timestamp или UUID).
    pub id: String,
    /// Относительный путь от корня проекта (data/gallery/sfw/001.png).
    pub path: PathBuf,
    /// Промпт, которым генерировалась картинка.
    pub prompt: String,
    /// Оценка vision-модели (0.0-10.0).
    pub score: f32,
    /// Категория: "sfw" или "nsfw".
    pub category: String,
    /// Unix timestamp создания.
    pub timestamp: u64,
}

/// Список всех фото в gallery.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GalleryMetadata {
    pub photos: Vec<GalleryPhoto>,
}

impl GalleryMetadata {
    /// Загрузить metadata.json (или создать пустой, если файл не существует).
    pub async fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let data = fs::read_to_string(path)
            .await
            .context("read metadata.json")?;
        serde_json::from_str(&data).context("parse metadata.json")
    }

    /// Сохранить metadata.json.
    pub async fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self).context("serialize metadata")?;
        fs::write(path, json).await.context("write metadata.json")
    }

    /// Добавить фото в метаданные.
    pub fn add(&mut self, photo: GalleryPhoto) {
        self.photos.push(photo);
    }

    /// Найти подходящее фото по категории (sfw/nsfw).
    /// Возвращает рандомное из доступных с score ≥ min_score.
    pub fn find_random(&self, category: &str, min_score: f32) -> Option<&GalleryPhoto> {
        let candidates: Vec<_> = self
            .photos
            .iter()
            .filter(|p| p.category == category && p.score >= min_score)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        candidates.choose(&mut rand::thread_rng()).copied()
    }

    /// Количество фото в категории.
    pub fn count(&self, category: &str) -> usize {
        self.photos.iter().filter(|p| p.category == category).count()
    }
}

/// Инициализация структуры папок gallery.
pub async fn init_gallery_dirs() -> Result<()> {
    fs::create_dir_all("data/gallery/sfw")
        .await
        .context("create data/gallery/sfw")?;
    fs::create_dir_all("data/gallery/nsfw")
        .await
        .context("create data/gallery/nsfw")?;
    Ok(())
}

/// Путь к metadata.json.
pub fn metadata_path() -> PathBuf {
    PathBuf::from("data/gallery/metadata.json")
}

/// Сохранить фото в gallery после успешной генерации и оценки.
pub async fn save_photo_to_gallery(
    image_path: &Path,
    prompt: &str,
    score: f32,
    nsfw: bool,
) -> Result<GalleryPhoto> {
    let category = if nsfw { "nsfw" } else { "sfw" };
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let id = format!("{}", timestamp);
    let filename = format!("{}_{}.png", category, id);
    let gallery_path = PathBuf::from(format!("data/gallery/{}/{}", category, filename));

    // Копируем сгенерированное фото в gallery
    fs::copy(image_path, &gallery_path)
        .await
        .context("copy image to gallery")?;

    let photo = GalleryPhoto {
        id,
        path: gallery_path,
        prompt: prompt.to_string(),
        score,
        category: category.to_string(),
        timestamp,
    };

    // Обновляем metadata.json
    let meta_path = metadata_path();
    let mut metadata = GalleryMetadata::load(&meta_path).await?;
    metadata.add(photo.clone());
    metadata.save(&meta_path).await?;

    tracing::info!(
        "[gallery] сохранено: {} (score={:.1}, category={})",
        prompt,
        score,
        category
    );

    Ok(photo)
}

/// Попытка взять фото из gallery (мгновенно, без SD).
pub async fn try_get_from_gallery(nsfw: bool, min_score: f32) -> Result<Option<PathBuf>> {
    let meta_path = metadata_path();
    let metadata = GalleryMetadata::load(&meta_path).await?;

    let category = if nsfw { "nsfw" } else { "sfw" };
    if let Some(photo) = metadata.find_random(category, min_score) {
        if photo.path.exists() {
            tracing::info!(
                "[gallery] взято из кэша: {} (score={:.1})",
                photo.prompt,
                photo.score
            );
            return Ok(Some(photo.path.clone()));
        } else {
            tracing::warn!(
                "[gallery] фото в metadata, но файл не найден: {:?}",
                photo.path
            );
        }
    }

    Ok(None)
}
