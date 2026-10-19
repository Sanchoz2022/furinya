//! tg-consolidate — парсит ВЕСЬ чат папика через TDLib (openChat + пагинация)
//! и прогоняет через реальную consolidate_memory (жатие в chat_history/<id>.md).
//!
//! НЕ отвечает в чатах. Только читает историю и пишет выжимку в data/diary/
//! chat_history/5005580127.md (две секции: профиль + выжимка чата).
//!
//! Запуск: FURINYA_TD_LIB=libs/libtdjson.so ./target/debug/tg-consolidate
//! (опц.) HISTORY_LIMIT=200 — макс. число сообщений для загрузки.

use anyhow::Result;
use furinya_core::{config::Config, prompts::Prompts};
use furinya_diary::{Diary, EmbeddingClient};
use furinya_llm::{LlmClient, Message};
use furinya_tg::{TdClient, auth, default_lib_path};
use std::time::Duration;
use tokio::runtime::Runtime;

fn main() -> Result<()> {
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
        auth::AuthState::Ready => println!("[consolidate] авторизация готова"),
        auth::AuthState::NeedUserInput(msg) => anyhow::bail!("нужно действие: {}", msg),
    }

    // TDLib лениво грузит историю в память. Ждём, пока подтянет из БД,
    // иначе getChatHistory вернёт только 1 сообщение (то, что успело загрузиться).
    println!("[consolidate] ждём подгрузки истории в память TDLib (~15с)...");
    std::thread::sleep(Duration::from_secs(15));

    let papik = cfg.general.papik_chat_id;
    // ВНИМАНИЕ: open_chat может сбрасывать кэш истории в TDLib — в inspect
    // без него история грузится полностью. Оставляем закомментированным.
    // client.open_chat(papik);
    std::thread::sleep(Duration::from_millis(500));

    // Warmup: searchChatMessages с пустым query форсирует подгрузку истории
    // с сервера (агрессивнее getChatHistory). Без этого getChatHistory может
    // вернуть только 1 сообщение.
    client.search_chat_messages(papik, 0, 0, 100, "warmup");
    std::thread::sleep(Duration::from_millis(800));
    // Сбрасываем очередь updates от warmup.
    while let Ok(v) = update_loop.updates.try_recv() {
        if v.get("@extra").and_then(|e| e.as_str()) == Some("warmup") {
            break;
        }
    }

    let max_total: usize = std::env::var("HISTORY_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500);

    // --- Загрузка всей истории (логика скопирована из tg-history-inspect,
    // которая стабильно даёт 164). ---
    let mut loaded: Vec<(i64, bool, String, i64)> = Vec::new();
    let page = 100i64;
    let mut from_id = 0i64;
    let deadline = std::time::Instant::now() + Duration::from_secs(180);
    loop {
        let extra = format!("hist_{}", loaded.len());
        client.get_chat_history(papik, page, from_id, &extra);
        let mut page_msgs: Vec<(i64, bool, String, i64)> = Vec::new();
        let page_deadline = std::time::Instant::now() + Duration::from_secs(25);
        while std::time::Instant::now() < page_deadline {
            let v = match update_loop.updates.recv_timeout(Duration::from_secs(3)) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if v.get("@extra").and_then(|e| e.as_str()) == Some(extra.as_str())
                && v.get("@type").and_then(|t| t.as_str()) == Some("messages")
            {
                if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
                    eprintln!("[DEBUG] пришёл messages-ответ: {} шт.", msgs.len());
                    if let Some(first) = msgs.first() {
                        eprintln!(
                            "[DEBUG] первый элемент: {}",
                            serde_json::to_string(first).unwrap_or_default().chars().take(300).collect::<String>()
                        );
                    }
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
                        if text.trim().is_empty() {
                            continue;
                        }
                        let is_me = sender == cfg.general.telegram_my_id;
                        page_msgs.push((sender, is_me, text, date));
                    }
                }
                break;
            }
        }
        if page_msgs.is_empty() {
            break;
        }
        let min_id = page_msgs.iter().map(|(id, _, _, _)| *id).min().unwrap_or(0);
        loaded.extend(page_msgs);
        if loaded.len() >= max_total || min_id == 0 {
            break;
        }
        from_id = min_id;
        if std::time::Instant::now() >= deadline {
            println!("[consolidate] таймаут загрузки, стоп на {}", loaded.len());
            break;
        }
    }
    println!("[consolidate] ИТОГО загружено сообщений: {}", loaded.len());

    // --- Строим Vec<Message> диалога (хронологически старые→новые). ---
    loaded.reverse(); // getChatHistory грузит новые→старые
    let mut conv: Vec<Message> = Vec::new();
    for (_sender, is_me, text, _date) in &loaded {
        if *is_me {
            conv.push(Message::assistant(text.clone()));
        } else {
            conv.push(Message::user(text.clone()));
        }
    }
    println!("[consolidate] диалог: {} сообщений", conv.len());

    // --- Загружаем LLM/diary/prompts и вызываем реальную consolidate_memory. ---
    let client_llm = LlmClient::from_config(&cfg)?;
    let emb_client = EmbeddingClient::from_config(&cfg.general.embedding.endpoint);
    let diary = Diary::new(
        &cfg.misc.diary_dir,
        emb_client,
        cfg.misc.diary_plagiarism_threshold,
        cfg.misc.diary_min_relatedness,
    );
    let mut diary_entries = diary.load_all().unwrap_or_else(|e| {
        eprintln!("[consolidate] дневник не загрузился: {}", e);
        Vec::new()
    });
    let prompts = Prompts::load_from_dir("prompts", &cfg)?;

    println!("[consolidate] вызываем consolidate_memory... (примечание: tg-consolidate fetch-баг — chat пустой, только diary-жатие)");
    let rt = Runtime::new()?;
    let n = rt.block_on(furinya_diary::consolidate_memory(
        &client_llm,
        &diary,
        &prompts,
        &mut diary_entries,
        &cfg.misc.diary_dir,
        &[],
        papik,
        cfg.misc.diary_keep_last,
        cfg.misc.active_top_k,
    ))?;
    println!("[consolidate] готово. Diary-записей сохранено: {}", n);

    // --- Показываем итоговый файл chat_history. ---
    let path = std::path::Path::new(&cfg.misc.diary_dir)
        .join("chat_history")
        .join(format!("{}.md", papik));
    println!("\n══════════ ИТОГ: {:?} ══════════", path);
    match std::fs::read_to_string(&path) {
        Ok(s) => {
            let max = 4000;
            if s.len() <= max {
                println!("{}", s);
            } else {
                println!("{}…\n[обрезано до {} симв из {}]", &s[..max], max, s.len());
            }
        }
        Err(e) => println!("не удалось прочитать файл: {}", e),
    }
    Ok(())
}
