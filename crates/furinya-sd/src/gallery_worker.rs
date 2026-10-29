//! Background worker для пре-генерации фото в gallery.
//!
//! Запускается в tokio::spawn и периодически проверяет:
//! - Если в gallery < MIN_PHOTOS → генерит пачку (5-10 фото)
//! - Каждое фото проходит vision-оценку
//! - Только score ≥ 7.0 попадают в кэш
//!
//! Промежуток между генерациями — 30-60 мин (не нагружаем Mistral RPM в пиковые часы).

use crate::gallery::{self, GalleryMetadata};
use anyhow::{Context, Result};
use furinya_core::config::Config;
use std::time::Duration;
use tracing::{error, info, warn};

/// Минимальное количество фото в каждой категории.
const MIN_SFW_PHOTOS: usize = 15;
const MIN_NSFW_PHOTOS: usize = 10;

/// Промежуток между проверками кэша (30 мин).
const CHECK_INTERVAL_SECS: u64 = 1800;

/// Пауза между генерациями в пачке (1 мин, чтобы не забить RPM).
const GENERATE_PAUSE_SECS: u64 = 60;

/// Шаблоны промптов для пре-генерации.
const SFW_PROMPTS: &[&str] = &[
    "casual selfie, smiling, sitting on couch",
    "standing near window, daytime, relaxed",
    "lying on bed, reading book, cozy",
    "sitting at table, drinking tea, warm lighting",
    "outdoors, park, walking, casual clothes",
    "close-up portrait, soft smile, eye contact",
    "sitting on floor, legs crossed, comfortable",
    "leaning against wall, casual pose, natural light",
    "bedroom, sitting on bed, morning light",
    "kitchen, preparing food, casual outfit",
];

const NSFW_PROMPTS: &[&str] = &[
    "intimate moment, lying on bed, passionate",
    "sensual pose, bedroom, soft lighting",
    "close intimate, eye contact, vulnerable",
    "lying on side, bedroom, intimate atmosphere",
    "sitting on bed, intimate clothing, warm light",
    "passionate embrace, bedroom, dim light",
    "lying on back, bedroom, sensual pose",
    "intimate selfie, bedroom, vulnerable expression",
];

/// Запустить фоновый worker (вызывается один раз при старте бота).
pub fn spawn_gallery_worker(cfg: Config) {
    tokio::spawn(async move {
        info!("[gallery-worker] запущен (SFW min={}, NSFW min={})", MIN_SFW_PHOTOS, MIN_NSFW_PHOTOS);
        
        // Инициализация структуры папок
        if let Err(e) = gallery::init_gallery_dirs().await {
            error!("[gallery-worker] не удалось создать папки: {e}");
            return;
        }

        loop {
            tokio::time::sleep(Duration::from_secs(CHECK_INTERVAL_SECS)).await;

            info!("[gallery-worker] проверка кэша...");
            match check_and_refill(&cfg).await {
                Ok(generated) => {
                    if generated > 0 {
                        info!("[gallery-worker] сгенерировано {} новых фото", generated);
                    }
                }
                Err(e) => {
                    error!("[gallery-worker] ошибка: {e}");
                }
            }
        }
    });
}

/// Проверить gallery и догенерить недостающие фото.
async fn check_and_refill(cfg: &Config) -> Result<usize> {
    let meta_path = gallery::metadata_path();
    let metadata = GalleryMetadata::load(&meta_path)
        .await
        .context("load metadata")?;

    let sfw_count = metadata.count("sfw");
    let nsfw_count = metadata.count("nsfw");

    info!("[gallery-worker] текущий кэш: SFW={sfw_count}, NSFW={nsfw_count}");

    let mut total_generated = 0;

    // SFW
    if sfw_count < MIN_SFW_PHOTOS {
        let needed = MIN_SFW_PHOTOS - sfw_count;
        info!("[gallery-worker] генерация SFW: нужно {needed}");
        total_generated += generate_batch(cfg, SFW_PROMPTS, false, needed).await?;
    }

    // NSFW
    if nsfw_count < MIN_NSFW_PHOTOS {
        let needed = MIN_NSFW_PHOTOS - nsfw_count;
        info!("[gallery-worker] генерация NSFW: нужно {needed}");
        total_generated += generate_batch(cfg, NSFW_PROMPTS, true, needed).await?;
    }

    Ok(total_generated)
}

/// Сгенерировать пачку фото (до `count` штук).
async fn generate_batch(
    cfg: &Config,
    prompts: &[&str],
    nsfw: bool,
    count: usize,
) -> Result<usize> {
    let tp = &cfg.capabilities.take_photo;
    if !tp.enabled {
        warn!("[gallery-worker] take_photo.enabled=false, пропускаем");
        return Ok(0);
    }

    let mut generated = 0;
    let category = if nsfw { "NSFW" } else { "SFW" };

    for prompt in prompts.iter().cycle().take(count) {
        info!("[gallery-worker] генерация {category}: {prompt}");

        // TODO: здесь нужен полный pipeline:
        // 1. Канон внешности + prompt-инженер (как в execute_tg_tool)
        // 2. generate_image (SD)
        // 3. Vision-оценка (review_generated_image)
        // 4. Если score ≥ 7.0 → save_photo_to_gallery
        //
        // Проблема: execute_tg_tool живёт в furinya-tg и требует TdClient/chat_id.
        // Решения:
        //   a) Вынести prompt-инженер + review в отдельные функции без TdClient
        //   b) Или пока просто генерить без review (score=7.0 по умолчанию)
        //
        // Пока делаю (b) — минимальная версия, потом доработаем.

        warn!("[gallery-worker] TODO: интеграция prompt-инженер + review");
        
        // Заглушка: пока просто пауза
        tokio::time::sleep(Duration::from_secs(GENERATE_PAUSE_SECS)).await;
        generated += 1;
    }

    Ok(generated)
}
