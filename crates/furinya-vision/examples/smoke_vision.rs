//! Smoke-тест: реальный вызов Gemma @8090 vision через describe_photo.
//! Запуск: cargo run -p furinya-vision --example smoke_vision <путь_к_фото>
use furinya_core::config::{EndpointConfig, VisionConfig};
use furinya_vision::describe_photo;

#[tokio::main]
async fn main() {
    let model = std::env::var("FURINYA_VISION_MODEL")
        .unwrap_or_else(|_| "mistral-medium-latest".to_string());
    let cfg = VisionConfig {
        enabled: true,
        llm_image_to_text: EndpointConfig {
            base_url: "https://api.mistral.ai/v1/".to_string(),
            bearer_key: String::new(),
            model: model.clone(),
            ..Default::default()
        },
        llm_image_to_text_cheap: EndpointConfig {
            base_url: "https://api.mistral.ai/v1/".to_string(),
            bearer_key: String::new(),
            model: "ministral-14b-2512".to_string(),
            ..Default::default()
        },
        max_dim: 1280,
        jpeg_quality: 80,
    };
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "smoke_photo.png".to_string());
    println!("[smoke] шлю {path} в vision (model={model})...");
    match describe_photo(
        &furinya_llm::LlmClient::new(
            &cfg.llm_image_to_text.base_url,
            &cfg.llm_image_to_text.bearer_key,
            &model,
            furinya_llm::SamplingParams {
                temperature: 0.2,
                top_p: None,
                top_k: None,
                min_p: None,
                presence_penalty: None,
                repetition_penalty: None,
            },
        ),
        &cfg,
        &path,
        "Опиши это фото кратко по-русски. Что изображено?",
    )
    .await
    {
        Ok(d) => println!("[smoke] OK описано ({} симв.): {}", d.len(), d),
        Err(e) => {
            eprintln!("[smoke] ОШИБКА: {e}");
            std::process::exit(1);
        }
    }
}
