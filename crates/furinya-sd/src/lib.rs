//! Генерация изображений (txt2img).
//!
//! Два бэкенда (выбирается в конфиге `take_photo.backend`):
//!   * `"aihorde"` — AI Horde / Stable Horde (https://aihorde.net/api/v2):
//!     краудсорс-ферма воркеров, БЕСПЛАТНО, работает NSFW (нужны воркеры с
//!     filter_sfw=false). Асинхронный протокол: POST /generate/async →
//!     poll /generate/check/{id} → GET /generate/status/{id} (картинка по URL с R2).
//!   * `"a1111"` — локальный AUTOMATIC1111 / Forge WebUI (`/sdapi/v1/txt2img`).
//!
//! Используется веткой take_photo бота: модель вызывает tool
//! `generate_image(prompt)`, мы стучимся в бэкенд, сохраняем PNG на диск
//! и возвращаем путь — дальше рантайм шлёт фото в Telegram.

use anyhow::{Context, Result};
use base64::Engine;
use furinya_core::config::{AiHordeConfig, EndpointConfig, SdRenderConfig, TakePhotoConfig};
use serde::Deserialize;
use std::time::{Duration, Instant};

/// Точка входа: выбирает бэкенд по конфигу. `nsfw` активно только на
/// локальном sd-server (Horde имеет собственный флаг cfg.aihorde.nsfw).
 pub async fn generate_image(
    cfg: &TakePhotoConfig,
    prompt: &str,
    negative_prompt: &str,
    nsfw: bool,
    out_path: &std::path::Path,
) -> Result<String> {
    let backend = cfg.backend.trim().to_lowercase();
    match backend.as_str() {
        "aihorde" | "horde" => {
            // Облако (SDXL-аниме: гетерохромия, NSFW, качество) — основной путь,
            // но очередь Horde 10-30 минут. Локальный sd-server (SD1.5) быстрый
            // (~2-3 мин). Поэтому гоним ОБА: кто первый — тот и ответ, но если
            // локал готов, даём облаку ещё CLOUD_GRACE_SECS — облачная картинка
            // качественнее, так что она приоритетнее.
            const CLOUD_GRACE_SECS: u64 = 600;
            let horde_path = out_path.with_extension("cloud.png");
            let local_path = out_path.with_extension("local.png");
            let horde = generate_image_aihorde(
                &cfg.aihorde,
                prompt,
                negative_prompt,
                nsfw,
                &horde_path,
            );
            let local = generate_image_a1111(
                &cfg.sd,
                &cfg.sd_render,
                prompt,
                negative_prompt,
                nsfw,
                &local_path,
            );
            tokio::pin!(horde);
            tokio::pin!(local);
            let (winner, _was_cloud) = tokio::select! {
                r = &mut horde => (r, true),
                r = &mut local => {
                    // Локальный готов: ещё 5 минут ждём облако.
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(CLOUD_GRACE_SECS),
                        &mut horde,
                    )
                    .await
                    {
                        Ok(r) => (r, true),
                        Err(_) => (r, false),
                    }
                }
            };
            let (src, is_cloud) = match &winner {
                Ok(_) => {
                    if horde_path.exists() && local_path.exists() {
                        // Оба успели — облачная лучше.
                        (horde_path.clone(), true)
                    } else if horde_path.exists() {
                        (horde_path.clone(), true)
                    } else {
                        (local_path.clone(), false)
                    }
                }
                Err(e) => {
                    tracing::warn!("[sd] облако не дало картинку ({e}) — откатываюсь на локальный sd-server");
                    match local.await {
                        Ok(p) => {
                            let _ = std::fs::rename(&local_path, out_path);
                            return Ok(p);
                        }
                        Err(e2) => anyhow::bail!("и облако, и локалный сервер не справились: {e} / {e2}"),
                    }
                }
            };
            let _ = std::fs::rename(&src, out_path);
            let _ = std::fs::remove_file(&horde_path);
            let _ = std::fs::remove_file(&local_path);
            tracing::info!(
                "[sd] фото готово ({}), {}",
                out_path.display(),
                if is_cloud { "облако SDXL" } else { "локальный SD1.5" }
            );
            Ok(out_path.display().to_string())
        }
        // Пустой backend и "sd" — локальный sd-server (наш основной).
        _ => {
            generate_image_a1111(
                &cfg.sd,
                &cfg.sd_render,
                prompt,
                negative_prompt,
                nsfw,
                out_path,
            )
            .await
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
// AI Horde (Stable Horde)
// ══════════════════════════════════════════════════════════════════════════

/// Horde отдаёт булевы поля то как `true/false`, то как `0/1`, то строкой
/// ("true"). Десериализатор-толерант: принимает всё это.
fn tolerant_bool<'de, D: serde::Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BoolLike {
        B(bool),
        I(i64),
        S(String),
    }
    Ok(match BoolLike::deserialize(d) {
        Ok(BoolLike::B(b)) => b,
        Ok(BoolLike::I(i)) => i != 0,
        Ok(BoolLike::S(s)) => matches!(s.trim().to_lowercase().as_str(), "true" | "1" | "yes"),
        Err(_) => false,
    })
}

/// То же для Option<bool> (поле может отсутствовать).
fn tolerant_bool_opt<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<bool>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BoolLike {
        N(()),
        B(bool),
        I(i64),
        S(String),
    }
    Ok(match BoolLike::deserialize(d) {
        Ok(BoolLike::N(())) => None,
        Ok(BoolLike::B(b)) => Some(b),
        Ok(BoolLike::I(i)) => Some(i != 0),
        Ok(BoolLike::S(s)) => Some(matches!(
            s.trim().to_lowercase().as_str(),
            "true" | "1" | "yes"
        )),
        Err(_) => None,
    })
}

/// Создание асинхронной задачи — ответ POST /generate/async.
#[derive(Deserialize)]
struct HordeAsyncResponse {
    id: String,
}

/// Ответ GET /generate/check/{id}.
#[derive(Deserialize)]
struct HordeCheck {
    #[serde(default, deserialize_with = "tolerant_bool")]
    processing: bool,
    #[serde(default, deserialize_with = "tolerant_bool")]
    waiting: bool,
    #[serde(default, deserialize_with = "tolerant_bool")]
    done: bool,
    #[serde(default, deserialize_with = "tolerant_bool")]
    faulted: bool,
    #[serde(default, deserialize_with = "tolerant_bool_opt")]
    is_possible: Option<bool>,
    #[serde(default)]
    queue_position: i64,
}

/// Ответ GET /generate/status/{id}.
#[derive(Deserialize)]
struct HordeStatus {
    generations: Vec<HordeGeneration>,
}

#[derive(Deserialize)]
struct HordeGeneration {
    /// URL картинки (R2-ссылка при r2=true) ИЛИ base64-тело при r2=false.
    img: String,
    #[serde(default, deserialize_with = "tolerant_bool")]
    censored: bool,
}

const AIHORDE_BASE: &str = "https://aihorde.net/api/v2";

/// Генерирует изображение через AI Horde и сохраняет PNG в `out_path`.
///
/// NSFW-запросы (`nsfw=true`, `censor_nsfw=false`) попадают только на
/// небезопасные воркеры — их меньше, поэтому очередь дольше. Анонимный
/// ключ "0000000000" работает, но приоритет минимальный.
pub async fn generate_image_aihorde(
    cfg: &AiHordeConfig,
    prompt: &str,
    negative_prompt: &str,
    nsfw: bool,
    out_path: &std::path::Path,
) -> Result<String> {
    // Ключ: env FURINYA_HORDE_KEY перекрывает конфиг (удобно / секреты вне config).
    let apikey = std::env::var("FURINYA_HORDE_KEY")
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .unwrap_or_else(|| cfg.apikey.clone());
    // NSFW-префикс — ТЕЖЕ, что и на локальном пути: без него SDXL-модель
    // рисует «взрослую» фигуру вместо компактной (проверено: f1 против f6).
    let prompt = if nsfw && !prompt.to_lowercase().contains("nsfw") {
        format!("nsfw, nude, flat chest, small breasts, petite, slim waist, slight body, {prompt}")
    } else {
        prompt.to_string()
    };
    // Негативный промпт в Horde вшивается в строку промпта через " ### ".
    let prompt_full = if negative_prompt.trim().is_empty() {
        prompt.to_string()
    } else {
        format!("{} ### {}", prompt.trim(), negative_prompt.trim())
    };

    let body = serde_json::json!({
        "prompt": prompt_full,
        "nsfw": cfg.nsfw,
        "censor_nsfw": cfg.censor_nsfw,
        "r2": true,
        "shared": false,
        "models": cfg.models,
        "params": {
            "width": cfg.width,
            "height": cfg.height,
            "steps": cfg.steps,
            "cfg_scale": cfg.cfg_scale,
            "sampler_name": cfg.sampler_name,
            "karras": true,
            // Horde API ждёт seed СТРОКОЙ ("-1" = случайный).
            "seed": "-1"
        }
    });
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .context("сборка HTTP-клиента для AI Horde")?;

    let resp = client
        .post(format!("{AIHORDE_BASE}/generate/async"))
        .header("apikey", &apikey)
        .header("Client-Agent", &cfg.client_agent)
        .json(&body)
        .send()
        .await
        .context("запрос generate/async к AI Horde")?;
    if resp.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
        anyhow::bail!("AI Horde: 429 — слишком много запросов к секретному/анонимному ключу, увеличь интервалы");
    }
    if !resp.status().is_success() {
        let status = resp.status();
        let body_txt = resp.text().await.unwrap_or_default();
        anyhow::bail!("AI Horde generate/async HTTP {status}: {body_txt}");
    }
    let job: HordeAsyncResponse = resp.json().await.context("парсинг ответа generate/async")?;
    let job_id = job.id;
    eprintln!("[sd][horde] задача создана: {job_id} — жду завершения…");

    // Поллинг: check до done/faulted/таймаут. Анонимный ключ в очереди может
    // стоять минуты — это нормально.
    let deadline = Instant::now() + Duration::from_secs(cfg.poll_timeout_secs.max(30));
    loop {
        if Instant::now() >= deadline {
            anyhow::bail!("AI Horde: таймаут ожидания {}с (задача {job_id}, очередь анонимам не резиновая. Может, зарегистрировать ключ на stablehorde.net/register?)", cfg.poll_timeout_secs);
        }
        tokio::time::sleep(Duration::from_millis(cfg.poll_interval_ms)).await;
        let check_res = client
            .get(format!("{AIHORDE_BASE}/generate/check/{job_id}"))
            .header("Client-Agent", &cfg.client_agent)
            .send()
            .await
            .context("poll generate/check")?;
        if !check_res.status().is_success() {
            anyhow::bail!(
                "AI Horde generate/check HTTP {}: {}",
                check_res.status(),
                check_res.text().await.unwrap_or_default()
            );
        }
        let check: HordeCheck = check_res.json().await.context("парсинг generate/check")?;
        if check.is_possible == Some(false) {
            anyhow::bail!("AI Horde: задача {job_id} невыполнима (is_possible=false) — параметры/модель недоступны");
        }
        if check.faulted {
            anyhow::bail!("AI Horde: задача {job_id} упала на воркерах (faulted)");
        }
        if check.done {
            break;
        }
        eprintln!(
            "[sd][horde] очередь: pos={}, processing={}, waiting={} (t={:.0}с)",
            check.queue_position,
            check.processing,
            check.waiting,
            (Instant::now() - (deadline - Duration::from_secs(cfg.poll_timeout_secs))).as_secs_f32()
        );
    }

    let status_res = client
        .get(format!("{AIHORDE_BASE}/generate/status/{job_id}"))
        .header("Client-Agent", &cfg.client_agent)
        .send()
        .await
        .context("GET generate/status")?;
    if !status_res.status().is_success() {
        anyhow::bail!(
            "AI Horde generate/status HTTP {}: {}",
            status_res.status(),
            status_res.text().await.unwrap_or_default()
        );
    }
    let status: HordeStatus = status_res.json().await.context("парсинг generate/status")?;
    let generation = status
        .generations
        .first()
        .ok_or_else(|| anyhow::anyhow!("AI Horde: пустой generations[]"))?;
    if generation.censored {
        // Воркер с SFW-фильтром вернул чёрный квадрат вместо картинки.
        // Это брак, а не результат: считаем задачу неудачей, чтобы верхний
        // уровень откатился на локальный sd-server (он NSFW-совместим).
        anyhow::bail!("AI Horde: картинка цензурирована воркером (черный квадрат)");
    }

    // img — либо https://... (R2), либо base64 (r2=false).
    let bytes: Vec<u8> = if generation.img.starts_with("http") {
        let img_resp = client.get(&generation.img).send().await.context("скачивание R2-картинки")?;
        if !img_resp.status().is_success() {
            anyhow::bail!("скачивание картинки: HTTP {}", img_resp.status());
        }
        img_resp.bytes().await?.to_vec()
    } else {
        base64::engine::general_purpose::STANDARD
            .decode(&generation.img)
            .context("декод base64 из generate/status")?
    };

    // Horde отдаёт webp/jpeg/png (зависит от воркера). Telegram-каналу нужен
    // PNG (inputMessagePhoto безопасно ест png/jpeg), поэтому перекодируем
    // в PNG всегда через image crate.
    write_as_png(&bytes, out_path)?;
    Ok(out_path.display().to_string())
}

/// Пишет байты изображения как PNG (перекодировка, если пришло webp/jpeg).
fn write_as_png(bytes: &[u8], out_path: &std::path::Path) -> Result<()> {
    fn is_png(b: &[u8]) -> bool {
        b.len() > 8 && &b[..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    }
    if is_png(bytes) {
        std::fs::write(out_path, bytes)
            .with_context(|| format!("запись файла {}", out_path.display()))?;
        return Ok(());
    }
    let img = image::load_from_memory(bytes).context("перекодирование картинки в PNG")?;
    img.save_with_format(out_path, image::ImageFormat::Png)
        .with_context(|| format!("сохранение PNG {}", out_path.display()))?;
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════
// Локальный AUTOMATIC1111 / Forge (legacy-бэкенд)
// ══════════════════════════════════════════════════════════════════════════

/// Ответ SD `/sdapi/v1/txt2img`: массив base64-PNG (обычно 1 штука).
#[derive(Deserialize)]
struct Txt2ImgResponse {
    images: Vec<String>,
}

/// Генерирует изображение через ЛОКАЛЬНЫЙ SD (A1111/Forge) и сохраняет в `out_path`.
///
/// * `endpoint` — конфиг `take_photo.sd` (base_url).
/// * `prompt` — позитивный промпт (уже с учётом стиля от модели).
/// * `negative_prompt` — что исключить (размытость, артефакты и т.п.).
/// * `nsfw` — явный контент: furina-LoRA 0.65 (иначе навязывает закрытый
///   жакет) + теги nsfw/nude в начало промпта; SFW — 0.75.
///
/// Сервер: stable-diffusion.cpp (sd-server, Proxmox LXC), чекпоинт
/// `meinahentai` (SD1.5) зашит на стороне сервера. LoRA строго через корневой
/// массив "lora" — инлайн-теги `<lora:...>` сервер НЕ парсит. 8 шагов +
/// Hyper-SD LoRA (CPU-рендер ~3.5 минуты, экономим шаги).
pub async fn generate_image_a1111(
    endpoint: &EndpointConfig,
    render: &SdRenderConfig,
    prompt: &str,
    negative_prompt: &str,
    nsfw: bool,
    out_path: &std::path::Path,
) -> Result<String> {
    // env SD_API_URL всегда перекрывает config (спецификация §1).
    let base_url = match std::env::var("SD_API_URL") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => endpoint.base_url.clone(),
    };
    let url = normalize_txt2img_url(&base_url);

    let mut furina_multiplier = 0.75;
    let mut prompt_final = prompt.to_string();
    if nsfw {
        furina_multiplier = 0.65;
        // Теги в начало (вес тегов в SD1.5 падает к концу промпта).
        // «компактность» на SDXL/Illustrious работает только через flat chest +
        // slim waist + slight body: проверено вживую — «petite/small breasts»
        // без них модель рисует полную фигуру (тест f1/f2 против f6).
        if !prompt_final.contains("nsfw") {
            prompt_final = format!(
                "nsfw, nude, flat chest, small breasts, petite, slim waist, slight body, {prompt_final}"
            );
        }
    }

    // LoRA: из конфига. Если в конфиге пусто — берём исторический дефолт
    // (furina + Hyper-SD 8-steps), иначе фото «поедут».
    let loras: Vec<serde_json::Value> = if render.loras.is_empty() {
        vec![
            serde_json::json!({"path": "furina.safetensors", "multiplier": furina_multiplier}),
            serde_json::json!({"path": "Hyper-SD15-8steps-lora.safetensors", "multiplier": 0.85}),
        ]
    } else {
        render
            .loras
            .iter()
            .map(|l| serde_json::json!({"path": l.path, "multiplier": l.multiplier}))
            .collect()
    };

    let body = serde_json::json!({
        "prompt": prompt_final,
        "negative_prompt": negative_prompt,
        "lora": loras,
        "steps": render.steps,
        "cfg_scale": render.cfg_scale,
        "width": render.width,
        "height": render.height,
        "sampler_name": render.sampler_name,
        "clip_skip": render.clip_skip,
        "seed": render.seed
    });

    // Спецификация §3.2: CPU-сервер никогда не должен получать 2 параллельных
    // рендера — глобальный семафор на все вызовы txt2img.
    static SD_SERVER_SEM: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);
    let _permit = SD_SERVER_SEM.acquire().await;

    let client = reqwest::Client::builder()
        // Рендер SD занимает ~4 минуты (замер: 237с) — таймаут с запасом,
        // иначе фото отваливается по таймауту под нагрузкой.
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("сборка HTTP-клиента для SD")?;

    // POST + парсинг; при ошибке "invalid lora path" прогреваем lora_cache
    // (GET /sdapi/v1/loras — спецификация §3.3) и ретраим один раз.
    let mut attempts = 0;
    let parsed: Txt2ImgResponse = loop {
        attempts += 1;
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("запрос txt2img к {url}"))?;
        let status = resp.status();
        if !status.is_success() {
            let txt = resp.text().await.unwrap_or_default();
            if attempts < 2 && (txt.contains("lora") || txt.contains("LoRA")) {
                let _ = warm_loras_cache(&client, &base_url).await;
                continue;
            }
            anyhow::bail!("SD HTTP {status}: {txt}");
        }
        break resp
            .json()
            .await
            .context("парсинг ответа SD (ожидался JSON с полем images[])")?;
    };
    let b64 = parsed
        .images
        .first()
        .ok_or_else(|| anyhow::anyhow!("SD вернул пустой images[]"))?;

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .context("декод base64 из ответа SD")?;
    std::fs::write(out_path, &bytes)
        .with_context(|| format!("запись файла {}", out_path.display()))?;
    Ok(out_path.display().to_string())
}

/// Превращает `http://host:port/` в корректный `/sdapi/v1/txt2img`.
fn normalize_txt2img_url(base: &str) -> String {
    let trimmed = base.trim_end_matches('/');
    format!("{trimmed}/sdapi/v1/txt2img")
}

/// Прогрев lora_cache sd-server (GET /sdapi/v1/loras): без него сервер может
/// выдавать "invalid lora path" на первом запросе (спецификация §3.3).
/// Дергается и при старте бота, и при лора-ошибке в генерации.
async fn warm_loras_cache(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<()> {
    let url = format!("{}/sdapi/v1/loras", base_url.trim_end_matches('/'));
    client
        .get(&url)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
        .context("GET /sdapi/v1/loras (прогрев lora_cache)")?;
    Ok(())
}

/// Прогрев lora_cache со Стартом бота — публичный для agent::run_telegram_mode.
/// Неблокирующий по сути (быстрый GET); при недоступном сервере не паникуем.
pub async fn warmup_sd_server(cfg: &TakePhotoConfig) {
    let _ = reqwest::Client::new()
        .get(format!("{}/sdapi/v1/loras", cfg.sd.base_url.trim_end_matches('/')))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await;
}

/// Live-интеграционный тест Horde: реально дёргает AI Horde и проверяет, что
/// PNG записан на диск. Запускается только с `HORDE_LIVE=1` (не входит в
/// обычный `cargo test`, чтобы не блокировать CI при выключенной сети).
///
/// Ключ берётся из env `FURINYA_HORDE_KEY` (регистрованный — приоритет выше),
/// иначе анонимный `0000000000`. Артефакт остаётся в `data/generated/` —
/// можно посмотреть руками.
#[tokio::test]
#[ignore]
async fn live_aihorde_txt2img_writes_png() {
    if std::env::var("HORDE_LIVE").is_err() {
        eprintln!("пропущен: задай HORDE_LIVE=1 для проверки AI Horde");
        return;
    }
    if std::env::var("FURINYA_HORDE_KEY").ok().filter(|k| !k.is_empty()).is_some() {
        eprintln!("[sd][horde] ключ: ЗАРЕГИСТРИРОВАННЫЙ (env FURINYA_HORDE_KEY) — приоритет в очереди");
    } else {
        eprintln!("[sd][horde] ключ: анонимный 0000000000 (задай FURINYA_HORDE_KEY для приоритета)");
    }
    let cfg = AiHordeConfig::default();
    // Куда сохранять: рабочая копия в data/generated/(workspace root), не /tmp.
    let root = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf();
    let out_dir = root.join("data/generated");
    let _ = std::fs::create_dir_all(&out_dir);
    let out = out_dir.join("aihorde_test.png");
    let _ = std::fs::remove_file(&out);
    let path = generate_image_aihorde(
        &cfg,
        "cute catgirl portrait, aqua blue short hair, blue-white frilly dress, anime art, soft studio lighting, masterpiece",
        "blurry, lowres, extra fingers, deformed hands, watermark",
        false,
        &out,
    )
    .await
    .expect("генерация не удалась");
    let meta = std::fs::metadata(&path).expect("файл не записан");
    assert!(meta.len() > 1000, "файл подозрительно мал: {} байт", meta.len());
    let sig = [0x89u8, 0x50, 0x4Eu8 as u8, 0x47];
    let head = std::fs::read(&path).unwrap();
    assert_eq!(&head[..4], &sig, "не PNG-сигнатура");
    eprintln!(
        "[sd][horde] ГОТОВО: {} ({} КБ) — открой и посмотри",
        path, meta.len() / 1024
    );
    // Артефакт НЕ удаляем: это тестовая картинка для визуальной проверки.
}
