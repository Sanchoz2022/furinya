//! Диагностика статуса online: принимает ли TDLib 1.8.62 опцию "online".
//! cargo run -p furinya-tg --example probe_online
//! (главный бот должен быть остановлен)

use anyhow::Context as _;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
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
        anyhow::bail!("не авторизован");
    }

    // 1) текущее значение опции
    let before = client.get_option_async("online").await?;
    println!("[probe] getOption(online) ДО: {}", before);

    // 2) ставим false
    client.set_online(false);
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    // 3) читаем снова
    let after = client.get_option_async("online").await?;
    println!("[probe] getOption(online) ПОСЛЕ setOption(false): {}", after);

    // 4) ждём 20с — ловим updateOption, если TDLib сам переигрывает
    println!("[probe] жду 20с updateOption/updateUserFullInfo...");
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
    while tokio::time::Instant::now() < deadline {
        if let Ok(u) = update_loop.updates.recv_timeout(std::time::Duration::from_secs(1)) {
            let t = u["@type"].as_str().unwrap_or("");
            if t.contains("Option") || t.contains("Status") || t.contains("Error") {
                println!("[probe] update: {} = {}", t, u);
            }
        }
    }
    Ok(())
}
