//! Аудио-модуль Furinya:
//!
//! * STT (распознавание): OpenAI-совместимый `/v1/audio/transcriptions`.
//!   Провайдер — Groq Cloud (whisper-large-v3-turbo) или локальный
//!   faster-whisper-server — формат запроса одинаковый (multipart/form-data).
//! * TTS (синтез голоса Furinya): Fish Audio `POST /v1/tts` с reference_id
//!   голосовой модели (см. docs/voice-reference-guide.md).
//!
//! Медленный локальный Whisper ушёл в прошлое — Groq крутит turbo-модель
//! за секунды. TTS-ветка — Fish Audio (поддерживает русский, в отличие от
//! Mistral Voxtral TTS, у которого русского в списке языков нет).

use anyhow::{Context, Result};
use furinya_core::config::EndpointConfig;
use reqwest::multipart::{Form, Part};
use std::path::Path;
use std::time::Duration;

/// Распознаёт аудиофайл через Whisper-сервис.
///
/// * `endpoint` — конфиг `capabilities.hearing.llm_audio_to_text` (base_url +
///   bearer_key + model). `base_url` обычно `http://localhost:8000/v1/`,
///   итоговый путь `/v1/audio/transcriptions`.
/// * `audio_path` — скачанный ogg/mp3/m4a/... файл из TDLib.
/// * `lang` — опционально `"ru"` для принудительного языка (точнее на смешанном).
///
/// Возвращает чистый распознанный текст (без обёрток).
pub async fn transcribe_audio(
    endpoint: &EndpointConfig,
    audio_path: &Path,
    lang: Option<&str>,
) -> Result<String> {
    let url = normalize_transcriptions_url(&endpoint.base_url);
    let bytes = std::fs::read(audio_path)
        .with_context(|| format!("чтение аудиофайла {}", audio_path.display()))?;
    let filename = audio_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("voice.oga")
        .to_string();

    let part = Part::bytes(bytes)
        .file_name(filename)
        .mime_str("audio/ogg")
        .unwrap_or_else(|_| Part::bytes(std::fs::read(audio_path).unwrap_or_default()));

    let mut form = Form::new()
        .text("model", endpoint.model.clone())
        .part("file", part);
    if let Some(l) = lang {
        form = form.text("language", l.to_string());
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120)) // Groq turbo крутит 5-мин голосовые за секунды
        .build()
        .context("сборка HTTP-клиента для STT")?;

    let mut req = client.post(&url).multipart(form);
    // Ключ: env GROQ_API_KEY перекрывает конфиг (секреты лучше держать вне TOML).
    let bearer = std::env::var("GROQ_API_KEY")
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .unwrap_or_else(|| endpoint.bearer_key.clone());
    if !bearer.is_empty() {
        req = req.bearer_auth(&bearer);
    }

    let resp = req
        .send()
        .await
        .with_context(|| format!("запрос STT к {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("STT HTTP {status}: {body}");
    }
    let body: serde_json::Value = resp.json().await.context("парсинг ответа STT")?;
    // OpenAI-совместимый ответ: {"text": "..."} (иногда обёрнут в {"segments":...})
    let text = body
        .get("text")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_default();
    Ok(text.trim().to_string())
}

/// Приводим base_url к `.../audio/transcriptions` без дублей слэшей.
fn normalize_transcriptions_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    // base уже содержит /v1 (config: http://localhost:8000/v1/)
    if base.ends_with("/audio/transcriptions") {
        base.to_string()
    } else {
        format!("{base}/audio/transcriptions")
    }
}

// ══════════════════════════════════════════════════════════════════════════
// TTS: Fish Audio (голос Фурины)
// ══════════════════════════════════════════════════════════════════════════

/// Синтезирует речь голосом Furinya через Fish Audio и сохраняет MP3 в `out_path`.
///
/// * `fish` — конфиг `[capabilities.record_voice.fish]` (key + referenceId).
/// * `text` — чистый текст для озвучки (без эмодзи/служебных строк — их
///   вырезает вызывающий код).
///
/// Endpoint: `POST https://api.fish.audio/v1/tts` (application/json), ответ —
/// поток аудио-байтов (MP3). reference_id — UUID голосовой модели из Fish
/// Audio (создаётся в дашборде с референсом furina_ref.wav).
pub async fn synthesize_fish_voice(
    fish: &furinya_core::config::FishTtsConfig,
    text: &str,
    out_path: &std::path::Path,
) -> anyhow::Result<String> {
    // Ключ: env FISH_API_KEY перекрывает конфиг.
    let key = std::env::var("FISH_API_KEY")
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .unwrap_or_else(|| fish.key.clone());
    if key.trim().is_empty() {
        anyhow::bail!("Fish Audio: пустой key в [capabilities.record_voice.fish]");
    }
    // reference_id опционален: без него Fish берёт дефолтный голос модели
    // (нужно для smoke-тестов пайплайна). БОТ-путь (agent.rs) перед использованием
    // всё равно проверяет reference_id — голому боту нечего озвучивать клоном.

    let mut body = serde_json::json!({
        "text": text,
        "format": "mp3",
        "mp3_bitrate": fish.mp3_bitrate,
        "normalize": true,
        "latency": "normal",
        "chunk_length": 200,
        "temperature": fish.temperature,
        "top_p": fish.top_p
    });
    // reference_id — UUID голосовой модели из Fish Audio (клон Фурины).
    if !fish.reference_id.trim().is_empty() {
        body["reference_id"] = serde_json::Value::String(fish.reference_id.trim().to_string());
    }
    // Строка модели: "s2.1-pro-free" = бесплатный тариф. Пусто = не передаём
    // (но тогда Fish откатывается на платный s2.1-pro — не оставляй пустым).
    // Официальные сниппеты Fish передают model как HTTP-ЗАГОЛОВОК, а не
    // только JSON-полем — на заголовке опирается пейволл-логика API.
    if !fish.model.trim().is_empty() {
        body["model"] = serde_json::Value::String(fish.model.trim().to_string());
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(240))
        .build()
        .context("сборка HTTP-клиента для Fish Audio TTS")?;
    let mut builder = client
        .post("https://api.fish.audio/v1/tts")
        .bearer_auth(&key);
    if !fish.model.trim().is_empty() {
        builder = builder.header("model", fish.model.trim().to_string());
    }
    let resp = builder
        .json(&body)
        .send()
        .await
        .context("запрос /v1/tts к Fish Audio")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body_txt = resp.text().await.unwrap_or_default();
        anyhow::bail!("Fish Audio TTS HTTP {status}: {body_txt}");
    }
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.bytes().await.context("чтение тела TTS-ответа")?;
    // Провайдер умеет вернуть ошибку JSON-ом с 200 — детектируем по content-type.
    if ct.contains("application/json") {
        anyhow::bail!("Fish Audio вернул JSON вместо аудио: {}", String::from_utf8_lossy(&bytes));
    }
    std::fs::write(out_path, &bytes)
        .with_context(|| format!("запись файла {}", out_path.display()))?;
    Ok(out_path.display().to_string())
}

/// Синтез через OpenAI-совместимый `POST {url}/audio/speech`.
///
/// Работает со ЛЮБЫМ совместимым сервером, в том числе полностью бесплатным
/// локальным XTTS-v2 (openedai-speech / xtts2_api / coqui-tts-local-server):
///   * git: https://github.com/matatonic/openedai-speech (model tts-1-hd + клон
///     голоса через config/voice_to_speaker.yaml → voice="furinya")
///   * git: https://github.com/lukaszliniewicz/xtts2_api (voice upload via /v1/files)
/// XTTS-v2 поддерживает русский и клон с 6–11 сек референса (furina_ref.wav,
/// mono 22050/24000Hz 16-bit WAV — см. docs/voice-reference-guide.md).
/// Возвращает путь к сохранённому файлу.
pub async fn synthesize_speech_openai(
    openai: &furinya_core::config::OpenAiTtsConfig,
    text: &str,
    out_path: &std::path::Path,
) -> anyhow::Result<String> {
    if openai.url.trim().is_empty() {
        anyhow::bail!("OpenAI-совместимый TTS: пустой url в [capabilities.record_voice.openai]");
    }
    let url = format!("{}/audio/speech", openai.url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": if openai.model.trim().is_empty() { "tts-1-hd" } else { openai.model.trim() },
        "input": text,
        "voice": if openai.voice.trim().is_empty() { "alloy" } else { openai.voice.trim() },
        "response_format": "mp3",
        "speed": 1.0
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(240))
        .build()
        .context("сборка HTTP-клиента для /audio/speech")?;

    let mut req = client.post(&url).json(&body);
    if !openai.key.trim().is_empty() {
        req = req.bearer_auth(&openai.key);
    }
    let resp = req.send().await.context(format!("запрос TTS к {url}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body_txt = resp.text().await.unwrap_or_default();
        anyhow::bail!("/audio/speech HTTP {status}: {body_txt}");
    }
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.bytes().await.context("чтение тела TTS-ответа")?;
    if ct.contains("application/json") {
        anyhow::bail!(
            "сервер вернул JSON вместо аудио (проверь voice/model): {}",
            String::from_utf8_lossy(&bytes)
        );
    }
    std::fs::write(out_path, &bytes)
        .with_context(|| format!("запись файла {}", out_path.display()))?;
    Ok(out_path.display().to_string())
}
