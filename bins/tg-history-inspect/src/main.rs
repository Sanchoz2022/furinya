//! tg-history-inspect — read-only инспектор истории чата папика.
//!
//! Логинится в TDLib (через существующую сессию в data/tdlib_db), открывает
//! чат (openChat — обязательно, иначе TDLib не держит историю локально),
//! пагинированно грузит историю (getChatHistory по from_message_id) и печатает:
//! - общее число загруженных сообщений
//! - временной диапазон (первое/последнее по дате)
//! - сэмплы (первые 5 и последние 5) с датами и отправителем
//!
//! НЕ отвечает в чатах, НЕ пишет ничего в память. Только чтение.

use furinya_core::config::Config;
use furinya_tg::{TdClient, auth, default_lib_path};
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let cfg = Config::load("config.toml")?;
    let lib_path = std::env::var("FURINYA_TD_LIB")
        .unwrap_or_else(|_| default_lib_path().to_string_lossy().into_owned());

    let (client, update_loop) = TdClient::start(
        &lib_path,
        cfg.general.telegram_api_id.into(),
        &cfg.general.telegram_api_hash,
        "data/tdlib_db",
    )?;

    match auth::authenticate(
        &client,
        cfg.general.telegram_api_id.into(),
        &cfg.general.telegram_api_hash,
        "data/tdlib_db",
        &update_loop.updates,
    )? {
        auth::AuthState::Ready => {
            println!("[inspect] авторизация готова, грузим историю...");
        }
        auth::AuthState::NeedUserInput(msg) => {
            anyhow::bail!("требуется действие пользователя: {}", msg);
        }
    }

    let papik = cfg.general.papik_chat_id;
    client.open_chat(papik);
    std::thread::sleep(Duration::from_millis(500));

    let max_total: i64 = std::env::var("HISTORY_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);

    let mut loaded: Vec<(i64, bool, String, i64)> = Vec::new();
    let page = 100i64;
    let mut from_id = 0i64;
    let deadline = std::time::Instant::now() + Duration::from_secs(120);

    loop {
        let extra = format!("hist_{}", loaded.len());
        client.get_chat_history(papik, page, from_id, &extra);
        // Ждём ответа с нашим extra.
        let mut page_msgs: Vec<(i64, bool, String, i64)> = Vec::new();
        let page_deadline = std::time::Instant::now() + Duration::from_secs(20);
        while std::time::Instant::now() < page_deadline {
            let v = match update_loop.updates.recv_timeout(Duration::from_secs(3)) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if v.get("@extra").and_then(|e| e.as_str()) == Some(extra.as_str())
                && v.get("@type").and_then(|t| t.as_str()) == Some("messages")
            {
                if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
                    for m in msgs {
                        let sender = m
                            .get("sender_id")
                            .and_then(|s| s.get("user_id"))
                            .and_then(|u| u.as_i64())
                            .unwrap_or(0);
                        let text = m
                            .get("content")
                            .and_then(|c| c.get("text"))
                            .and_then(|t| t.get("text"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string();
                        let date = m.get("date").and_then(|d| d.as_i64()).unwrap_or(0);
                        let id = m.get("id").and_then(|i| i.as_i64()).unwrap_or(0);
                        if text.trim().is_empty() {
                            continue;
                        }
                        page_msgs.push((id, sender == cfg.general.telegram_my_id, text, date));
                    }
                }
                break;
            }
        }
        if page_msgs.is_empty() {
            break; // больше нечего грузить
        }
        // Запоминаем минимальный id страницы для следующей итерации (старые сообщения).
        let min_id = page_msgs.iter().map(|(id, _, _, _)| *id).min().unwrap_or(0);
        loaded.extend(page_msgs);
        if loaded.len() as i64 >= max_total || min_id == 0 {
            break;
        }
        from_id = min_id;
        if std::time::Instant::now() >= deadline {
            println!("[inspect] таймаут загрузки, останавливаемся на {}", loaded.len());
            break;
        }
    }

    println!(
        "\n══════════ ИСТОРИЯ ЧАТА (папик={}) ══════════",
        papik
    );
    println!("Загружено сообщений: {}", loaded.len());
    if loaded.is_empty() {
        println!("(пусто — возможно, TDLib ещё не подгрузил или чат пуст)");
        return Ok(());
    }

    let dates: Vec<i64> = loaded.iter().map(|(_, _, _, d)| *d).collect();
    let min_d = dates.iter().min().unwrap_or(&0);
    let max_d = dates.iter().max().unwrap_or(&0);
    println!(
        "Диапазон дат: {} .. {} (unix)",
        chrono_fmt(*min_d),
        chrono_fmt(*max_d)
    );

    // Сохраняем ВЕСЬ чат в файл для последующей консолидации (file-consolidate).
    // Формат: "[ДД.ММ ЧЧ:ММ] Имя: текст" (одна строка на сообщение), хронологически
    // старые сверху. TDLib отдаёт новые->старые, поэтому реверсим loaded.
    let raw_path = std::path::Path::new(&cfg.misc.diary_dir)
        .join("chat_history")
        .join(format!("_raw_{}.txt", papik));
    if let Some(parent) = raw_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let bot_name = cfg.general.character_name.clone();
    let papik_name = cfg.general.papik_name.clone();
    let mut raw_content = String::new();
    for (_id, is_me, text, date) in loaded.iter().rev() {
        let who = if *is_me { &bot_name } else { &papik_name };
        let dt = chrono::DateTime::from_timestamp(*date, 0)
            .map(|d| d.format("%d.%m %H:%M").to_string())
            .unwrap_or_else(|| "??.?? ??:??".to_string());
        raw_content.push_str(&format!("[{}] {}: {}\n", dt, who, text));
    }
    match std::fs::write(&raw_path, &raw_content) {
        Ok(_) => println!("[inspect] сырой чат сохранён: {:?} ({} строк)", raw_path, loaded.len()),
        Err(e) => println!("[inspect] не удалось сохранить сырой чат: {}", e),
    }

    let my_id = cfg.general.telegram_my_id;
    let show = |label: &str, slice: &[(i64, bool, String, i64)]| {
        println!("\n── {} ({} шт.) ──", label, slice.len());
        for (sender, is_me, text, date) in slice {
            let who = if *is_me { "ФУРИНЬЯ" } else { "ПАПИК" };
            let _ = (sender, my_id);
            println!(
                "[{}] {}: {}",
                chrono_fmt(*date),
                who,
                text.chars().take(120).collect::<String>()
            );
        }
    };

    let n = loaded.len();
    let head = &loaded[..n.min(5)];
    let tail = if n >= 5 { &loaded[n - 5..] } else { &loaded[..] };
    show("ПЕРВЫЕ (старые)", head);
    show("ПОСЛЕДНИЕ (новые)", tail);

    println!("\n══════════════════════════════════════════════════════");
    Ok(())
}

/// Грубый формат unix-секунд в читаемую строку (без внешних зависимостей).
fn chrono_fmt(unix: i64) -> String {
    if unix == 0 {
        return "?".into();
    }
    let days = unix / 86400;
    let rem = unix % 86400;
    let hh = rem / 3600;
    let mm = (rem % 3600) / 60;
    let year = 1970 + days / 365;
    let doy = days % 365;
    format!("{}-{:03} {:02}:{:02}Z", year, doy, hh, mm)
}
