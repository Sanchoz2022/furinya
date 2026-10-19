//! Smoke-тест живого цикла ГОЛОСА (путь БОТА):
//!   1) Fish Audio TTS — секция `[capabilities.record_voice.fish]` из config.toml
//!      (referenceId клона Фурины + модель s2.1-pro-free), ключ из env `FISH_API_KEY`;
//!   2) Groq STT — `whisper-large-v3-turbo`, ключ из env `GROQ_API_KEY`.
//! Второй шаг тем самым проверяет, что синтез не отдал нам фальшивый mp3.
//!
//! Запуск: `cargo run -p furinya-audio --example smoke_fish`
//! Артефакты: data/generated/fish_test.mp3 (остаётся — музыка/RU-голос на слух).

use furinya_audio::{synthesize_fish_voice, transcribe_audio};
use furinya_core::config::{Config, EndpointConfig};
use std::path::Path;

#[tokio::main]
async fn main() {
    // Берём Fish-секцию из РЕАЛЬНОГО config.toml (уровень бота).
    let root = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf();
    let out_dir = root.join("data/generated");
    std::fs::create_dir_all(&out_dir).unwrap();
    let mp3 = out_dir.join("fish_test.mp3");

    let fish = root
        .join("config.toml")
        .to_str()
        .and_then(|p| Config::load(p).ok())
        .map(|c| c.capabilities.record_voice.fish)
        .unwrap_or_default();
    if fish.reference_id.trim().is_empty() {
        eprintln!("[fish] ВНИМАНИЕ: referenceId пуст в config.toml — синтез пойдёт ДЕФОЛТНЫМ голосом модели");
    } else {
        println!("[fish] клон Фурины (referenceId={})...", fish.reference_id);
    }
    let text = "Привет! Я Фуриня. Это проверка синтеза речи и распознавания. Мяу.";
    println!(
        "[fish] синтез '{}' (model={}, клон={}, {} байт)...",
        text,
        fish.model,
        if fish.reference_id.trim().is_empty() { "нет — дефолтный голос" } else { fish.reference_id.get(..8).unwrap_or("") },
        text.len()
    );
    match synthesize_fish_voice(&fish, text, &mp3).await {
        Ok(p) => {
            let bytes = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            println!("[fish] OK: {} ({} байт MP3)", p, bytes);
        }
        Err(e) => {
            eprintln!("[fish] ОШИБКА: {e}");
            std::process::exit(1);
        }
    }

    // Roundtrip: отдаём свежий mp3 в Groq whisper.
    let stt_cfg = EndpointConfig {
        base_url: "https://api.groq.com/openai/v1/".to_string(),
        bearer_key: String::new(), // env GROQ_API_KEY подхватится внутри transcribe_audio
        model: "whisper-large-v3-turbo".to_string(),
        ..Default::default()
    };
    println!("[groq] распознаю {} ...", mp3.display());
    match transcribe_audio(&stt_cfg, &mp3, Some("ru")).await {
        Ok(t) => println!("[groq] OK: {t:?}"),
        Err(e) => {
            eprintln!("[groq] ОШИБКА: {e}");
            std::process::exit(1);
        }
    }
}
