//! Smoke-тест: реальный вызов Whisper @8000 (проверяет HTTP-путь и формат ответа).
//! Запуск: cargo run -p furinya-audio --example smoke_audio
use furinya_audio::transcribe_audio;
use furinya_core::config::EndpointConfig;
use std::path::Path;

#[tokio::main]
async fn main() {
    let cfg = EndpointConfig {
        base_url: "http://localhost:8000/v1/".to_string(),
        bearer_key: String::new(),
        model: "whisper-large-v3-turbo".to_string(),
        ..Default::default()
    };
    // Тестовый ogg должен лежать рядом (генерируется скриптом/host).
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "smoke_voice.oga".to_string());
    let p = Path::new(&path);
    if !p.exists() {
        eprintln!("НЕТ файла {path} — сгенерируй: ffmpeg -f lavfi -i \"sine=frequency=440:duration=3\" -c:a libopus smoke_voice.oga");
        std::process::exit(2);
    }
    println!("[smoke] шлю {} в Whisper...", path);
    match transcribe_audio(&cfg, p, Some("ru")).await {
        Ok(t) => println!("[smoke] OK распознано ({} симв.): {:?}", t.len(), t),
        Err(e) => {
            eprintln!("[smoke] ОШИБКА: {e}");
            std::process::exit(1);
        }
    }
}
