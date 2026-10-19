//! Smoke-тест генерации ФОТО так же, как это делает бот:
//! берёт `[capabilities.take_photo]` из config.toml (backend=aihorde) и
//! дёргает `generate_image` — ту же функцию, что и tool `generate_image`.
//! Ключ Horde: env `FURINYA_HORDE_KEY` перекрывает config (это внутренняя
//! логика furinya-sd).
//!
//! Запуск: `cargo run -p furinya-sd --example smoke_horde`
//! Артефакт: data/generated/aihorde_smoke.png (остаётся).

use furinya_core::config::Config;
use furinya_sd::generate_image;
use std::path::Path;

#[tokio::main]
async fn main() {
    let root = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf();
    let out = root.join("data/generated/aihorde_smoke.png");
    let _ = std::fs::remove_file(&out);

    let cfg = Config::load(root.join("config.toml"))
        .unwrap_or_else(|e| panic!("config.toml не загрузился: {e}"));
    let photo = &cfg.capabilities.take_photo;
    println!(
        "[sd] enabled={}, backend={}, настройки Horde: {}x{}, steps={}, модели={:?}",
        photo.enabled,
        photo.backend,
        photo.aihorde.width,
        photo.aihorde.height,
        photo.aihorde.steps,
        photo.aihorde.models
    );
    if !photo.enabled {
        eprintln!("[sd] ВНИМАНИЕ: capabilities.take_photo.enabled=false — бот бы отказался");
    }
    println!("[sd] послал... ждём очередь Horde (регистр ключ выше приоритет)");

    let path = generate_image(
        photo,
        "cute catgirl, aqua blue short hair, blue-white frilly dress, soft warm candlelight ambiance, anime art, masterpiece",
        "blurry, lowres, extra fingers, deformed hands, watermark",
        false,
        &out,
    )
    .await
    .unwrap_or_else(|e| panic!("[sd] генерация не удалась: {e}"));

    let meta = std::fs::metadata(&path).unwrap();
    println!(
        "[sd] ГОТОВО: {path} ({} КБ) — посмотрите глазами",
        meta.len() / 1024
    );
}
