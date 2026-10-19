//! Одиночная утилита: ставит PNG/JPEG как аватарку аккаунта (setProfilePhoto).
//!
//! Использование: cargo run -p furinya-tg --example set_avatar -- <path.png>
//! ВАЖНО: главный бот должен быть ОСТАНОВЛЕН (td.binlog не должен быть занят).
//!
//! Запускает TDLib, дожидается Ready, шлёт setProfilePhoto, ждёт 8с, выходит.

use anyhow::Context as _;
use serde_json::Value;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("usage: set_avatar <path.png>")?;
    let path = std::fs::canonicalize(&path)?;
    println!("[avatar] ставлю аватарку: {}", path.display());

    let cfg = furinya_core::config::Config::load(std::path::Path::new("config.toml"))
        .context("config.toml")?;

    let (client, update_loop) = furinya_tg::TdClient::start(
        furinya_tg::default_lib_path().to_str().unwrap_or("libs/libtdjson.so"),
        cfg.general.telegram_api_id.into(),
        &cfg.general.telegram_api_hash,
        "data/tdlib_db",
    )?;

    let state = furinya_tg::auth::authenticate(
        &client,
        cfg.general.telegram_api_id.into(),
        &cfg.general.telegram_api_hash,
        "data/tdlib_db",
        &update_loop.updates,
    )?;
    if matches!(state, furinya_tg::auth::AuthState::NeedUserInput(_)) {
        anyhow::bail!("не авторизован — сначала запусти бота и выполни вход");
    }

    client.set_profile_photo(path.to_str().unwrap());
    println!("[avatar] setProfilePhoto отправлен, жду 8с...");
    tokio::time::sleep(std::time::Duration::from_secs(8)).await;

    // Смотрим, пришёл ли updateProfilePhoto / error.
    let mut checked = 0;
    while let Ok(u) = update_loop.updates.try_recv() {
        if let Some(t) = u["@type"].as_str() {
            println!("[avatar] update: {}", t);
        }
        checked += 1;
        if checked > 20 { break; }
    }

    client.send(&serde_json::json!({ "@type": "close" }));
    println!("[avatar] готово");
    Ok(())
}
