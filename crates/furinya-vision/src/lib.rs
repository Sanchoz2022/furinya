//! furinya-vision — анализ изображений от пользователя.
//!
//! Берёт локальный путь к скачанному из TDLib фото, при необходимости
//! даунскейлит (vision.max_dim) и перекодирует в JPEG (vision.jpeg_quality),
//! затем шлёт в multimodal-LLM (Gemma @8090) и возвращает текстовое описание.
//!
//! Никаких хардкодов размеров (как было в kuni: вшитые 64 → 1000+ токенов
//! на любую фотку). Все лимиты — в конфиге.

use anyhow::Context;
use base64::Engine as _;
use furinya_core::config::VisionConfig;
use furinya_llm::{ContentPart, ImageUrl, LlmClient, Message, MessageContent};
use image::codecs::jpeg::JpegEncoder;

/// Загружает картинку, даунскейлит до `max_dim` (если >0) и перекодирует
/// в JPEG с `quality`. Возвращает base64 data-URI готовый для LLM.
fn encode_image(path: &str, max_dim: u32, quality: u8) -> anyhow::Result<String> {
    // BY CONTENT, а не по расширению: Horde отдаёт WebP под именем *.png —
    // image::open() угадывает формат по расширению и падает. Читаем байты и
    // пускаем снюффер формата (PNG/JPEG/WebP/GIF — что угодно).
    let bytes = std::fs::read(path)
        .with_context(|| format!("не удалось прочитать файл изображения: {}", path))?;
    let img = image::load_from_memory(&bytes)
        .with_context(|| format!("не удалось декодировать изображение (по содержимому): {}", path))?;

    // Даунскейл по большей стороне (max_dim == 0 → пропускаем).
    let img = if max_dim > 0 {
        let (w, h) = (img.width(), img.height());
        let longest = w.max(h);
        if longest > max_dim {
            let scale = max_dim as f32 / longest as f32;
            let nw = (w as f32 * scale).max(1.0) as u32;
            let nh = (h as f32 * scale).max(1.0) as u32;
            img.resize(nw, nh, image::imageops::FilterType::Lanczos3)
        } else {
            img
        }
    } else {
        img
    };

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut enc = JpegEncoder::new_with_quality(&mut buf, quality);
        enc.encode(
            img.as_bytes(),
            img.width(),
            img.height(),
            img.color().into(),
        )
        .with_context(|| "не удалось закодировать JPEG")?;
    }

    Ok(format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&buf)
    ))
}

/// Описывает фото через multimodal-LLM.
///
/// `prompt` — инструкция на русском (что описать). Результат — текстовое
/// описание от модели (для вставки в контекст ответа бота).
pub async fn describe_photo(
    llm: &LlmClient,
    cfg: &VisionConfig,
    path: &str,
    prompt: &str,
) -> anyhow::Result<String> {
    let data_uri = encode_image(path, cfg.max_dim, cfg.jpeg_quality)?;

    let messages = vec![
        Message::system(
            "You are a vision assistant. Describe the image concisely in the same \
             language the user writes in (usually Russian). Focus on what matters \
             for a personal AI companion: people, objects, context, mood. \
             Do NOT invent details not visible. OUTPUT IN RUSSIAN.",
        ),
        Message::user_parts(vec![
            ContentPart::Text {
                text: prompt.to_string(),
            },
            ContentPart::Image {
                image_url: ImageUrl { url: data_uri },
            },
        ]),
    ];

    let resp = llm.chat(messages, None).await?;
    let text = match resp.choices.into_iter().next() {
        Some(c) => match c.message.content {
            MessageContent::Text(s) => s,
            MessageContent::Parts(_) => String::new(),
        },
        None => String::new(),
    };
    Ok(text.trim().to_string())
}
