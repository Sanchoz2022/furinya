//! Прогон РЕАЛЬНОГО хода агента на реальных API (Mistral/Gemini).
//!
//! Нужен, чтобы увидеть без Telegram, что происходит в проде: срабатывает ли
//! оценка хода, что уходит в system-промпт и какой текст модель реально шлёт.
//!
//! Запуск (на сервере, из /opt/furinya, с ключами из .env):
//!   replay <config.toml> <prompts_dir> <scenario.txt>
//!
//! Формат сценария, по строке на реплику:
//!   u|<unix_ts>|<текст>   — сообщение Александра (живой ход)
//!   m|<unix_ts>|<текст>   — уже отправленная реплика бота (контекст)
//!
//! Последняя строка `u|...` — то, на что отвечаем; предыдущие — контекст.
//!
//! `REPLAY_IDLE=1` — прогнать ВНУТРЕННИЙ idle-ход (он не писал, у неё шанс
//! написать первой): все строки сценария идут как контекст, текст stim-а
//! собирается тем же `idle_stimulus_text`, что и в проде.

use std::io::Write;
use std::path::PathBuf;

use furinya_core::Config;
use furinya_core::prompts::Prompts;
use furinya_diary::{Diary, EmbeddingClient};
use furinya_llm::{LlmClient, Message};
use furinya_tg::TdClient;
use furinya_tg::agent::{handle_new_message, idle_stimulus_text};
use serde_json::{Value, json};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cfg_path = args.get(1).map(String::as_str).unwrap_or("config.toml");
    let prompts_dir = args.get(2).map(String::as_str).unwrap_or("prompts");
    let scenario_path = args.get(3).map(String::as_str).unwrap_or("scenario.txt");

    // Логи — как в проде: stdout + файл.
    let _ = furinya_core::logging::init_logging("/tmp/furinya_replay.log");

    let mut cfg = Config::load(cfg_path)?;
    let work = match std::env::var("REPLAY_DIARY") {
        // Готовый снимок прод-памяти (RAG/профиль/выжимка) — качество ответа
        // сильно зависит от того, что она реально помнит.
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            let p = PathBuf::from("/tmp/furinya_replay_diary");
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p)?;
            p
        }
    };
    std::fs::create_dir_all(&work)?;
    cfg.misc.diary_dir = work.display().to_string();

    let prompts = Prompts::load_from_dir(prompts_dir, &cfg)?;
    let system_prompt = prompts.build_system_prompt();
    if let Ok(m) = std::env::var("REPLAY_MODEL") {
        cfg.general.llm.endpoint.model = m;
    }
    if let Ok(t) = std::env::var("REPLAY_TEMP") {
        if let Ok(v) = t.parse::<f32>() {
            cfg.misc.llm_temperature = v;
        }
    }
    let llm = LlmClient::from_config(&cfg)?;
    let diary = Diary::new(
        &cfg.misc.diary_dir,
        EmbeddingClient::from_config(&cfg.general.embedding.endpoint),
        cfg.misc.diary_plagiarism_threshold,
        cfg.misc.diary_min_relatedness,
    );
    let mut diary_entries = diary.load_all().unwrap_or_default();
    let (client, rx) = TdClient::test_client();

    let chat_id = cfg.general.papik_chat_id;
    let my_id = cfg.general.telegram_my_id;

    let raw = std::fs::read_to_string(scenario_path)?;
    let rows: Vec<(bool, i64, String)> = raw
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let mut it = l.splitn(3, '|');
            let who = it.next()?.trim();
            let ts = it.next()?.trim().parse::<i64>().ok()?;
            let text = it.next()?.to_string();
            Some((who == "m", ts, text))
        })
        .collect();
    anyhow::ensure!(!rows.is_empty(), "пустой сценарий");

    let idle_turn = std::env::var("REPLAY_IDLE")
        .map(|v| v == "1")
        .unwrap_or(false);

    let mut tg_history: Vec<Message> = vec![Message::system(system_prompt.clone())];
    let mut tg_history_full: Vec<(i64, bool, String, i64)> = Vec::new();
    let ctx_end = if idle_turn { rows.len() } else { rows.len() - 1 };
    for (is_me, ts, text) in &rows[..ctx_end] {
        tg_history_full.push((if *is_me { my_id } else { chat_id }, *is_me, text.clone(), *ts));
        let who = if *is_me {
            cfg.general.character_name.clone()
        } else {
            cfg.general.papik_name.clone()
        };
        tg_history.push(if *is_me {
            Message::assistant(format!("[{}] {}: {}", ts, who, text))
        } else {
            Message::user(format!("[{}] {}: {}", ts, who, text))
        });
    }
    // Живой ход: последняя строка — его сообщение. Idle: все строки контекст,
    // а текст стимула собираем ровно тем же кодом, что и прод.
    let (ts, text, sender) = if idle_turn {
        use chrono::Timelike;
        let last_ts = rows.last().map(|r| r.1).unwrap_or(0);
        let silent_minutes = ((chrono::Utc::now().timestamp() - last_ts).max(0) / 60) as u64;
        let now = chrono::Local::now();
        let self_state = std::fs::read_to_string(work.join("self_state.md")).unwrap_or_default();
        let self_state_block = {
            let s = self_state.trim();
            if s.is_empty() {
                String::new()
            } else {
                let cut: String = s.chars().take(800).collect();
                format!("\n\nТвоё текущее состояние (можешь опереться на него):\n{cut}")
            }
        };
        (
            now.timestamp(),
            idle_stimulus_text(
                now.hour(),
                now.minute(),
                silent_minutes,
                &self_state_block,
                "",
                "",
            ),
            0,
        )
    } else {
        let (is_me, ts, text) = rows.last().cloned().unwrap();
        anyhow::ensure!(!is_me, "последняя строка сценария должна быть его сообщением (u|...)");
        (ts, text, chat_id)
    };

    let incoming = json!({
        "message": {
            "chat_id": chat_id,
            "id": if idle_turn { 0 } else { 424242 },
            "date": ts,
            "sender_id": {"user_id": sender},
            "content": {
                "@type": "messageText",
                "text": {"@type": "formattedText", "text": text}
            }
        }
    });

    if idle_turn {
        println!("\n=== ХОД: idle (он не писал) ===\n{text}\n");
    } else {
        println!("\n=== ХОД: его сообщение ===\n{text}\n");
    }
    handle_new_message(
        incoming,
        &llm,
        &client,
        &diary,
        &prompts,
        &mut diary_entries,
        &mut tg_history,
        &mut tg_history_full,
        &system_prompt,
        &cfg,
        my_id,
        idle_turn,
    )
    .await;

    let mut sent: Vec<String> = Vec::new();
    while let Ok(Some(cmd)) = rx.try_recv() {
        let Ok(v) = serde_json::from_str::<Value>(&cmd) else {
            continue;
        };
        if v["@type"].as_str() == Some("sendMessage") {
            sent.push(
                v["input_message_content"]["text"]["text"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
            );
        }
    }
    println!("\n=== ОТПРАВЛЕНО В ЧАТ ({} частей) ===", sent.len());
    for (i, s) in sent.iter().enumerate() {
        println!("[{i}] {s}");
    }

    let state = work.join("chat_state").join(format!("{chat_id}.md"));
    println!("\n=== chat_state ===");
    match std::fs::read_to_string(&state) {
        Ok(s) => println!("{s}"),
        Err(e) => println!("(нет: {e})"),
    }
    let sit = work.join("situation").join(format!("{chat_id}.md"));
    println!("\n=== situation ===");
    match std::fs::read_to_string(&sit) {
        Ok(s) => println!("{s}"),
        Err(e) => println!("(нет: {e})"),
    }
    let _ = std::io::stdout().flush();
    Ok(())
}
