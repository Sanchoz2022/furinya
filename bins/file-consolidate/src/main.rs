//! file-consolidate — читает сохранённый сырой чат (tg-history-inspect ->
//! data/diary/chat_history/_raw_<chat_id>.txt) и прогоняет через реальную
//! consolidate_memory (жатие в chat_history/<id>.md через LLM :8090).
//!
//! НЕ трогает TDLib. Только читает файл и пишет выжимку.
//!
//! Запуск: ./target/debug/file-consolidate
//! (опц.) CHAT_ID=5005580127 — какой файл _raw_<id>.txt брать.

use anyhow::Result;
use furinya_core::{config::Config, prompts::Prompts};
use furinya_diary::{Diary, EmbeddingClient};
use furinya_llm::LlmClient;
use std::path::PathBuf;
use tokio::runtime::Runtime;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let cfg = Config::load("config.toml")?;
    let chat_id: i64 = std::env::var("CHAT_ID")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(cfg.general.papik_chat_id);

    let raw_path: PathBuf = std::path::Path::new(&cfg.misc.diary_dir)
        .join("chat_history")
        .join(format!("_raw_{}.txt", chat_id));
    println!("[file-consolidate] читаем сырой чат: {:?}", raw_path);
    let raw = std::fs::read_to_string(&raw_path)
        .map_err(|e| anyhow::anyhow!("не удалось прочитать {:?}: {}", raw_path, e))?;

    // Парсим строки "[ДД.ММ ЧЧ:ММ] Имя: текст" в Vec<ChatTurn>.
    // Отсекаем служебный мусор (команды консолидации и ответы бота о ней),
    // чтобы он не попадал в выжимку как "диалог".
    let mut conv: Vec<furinya_diary::ChatTurn> = Vec::new();
    let re = regex::Regex::new(r"^\[([^\]]+)\]\s*([^:]+):\s*(.*)$").unwrap();
    for line in raw.lines() {
        if let Some(caps) = re.captures(line) {
            let who = caps.get(2).unwrap().as_str().trim().to_string();
            let text = caps.get(3).unwrap().as_str().trim().to_string();
            let is_service = text.trim() == "!consolidate"
                || text.contains("Консолидация выполнена")
                || text.contains("✅");
            if is_service {
                continue;
            }
            conv.push(furinya_diary::ChatTurn {
                date: caps.get(1).unwrap().as_str().to_string(),
                who,
                text,
            });
        }
    }
    println!("[file-consolidate] распарсено сообщений: {}", conv.len());
    if conv.len() < 2 {
        anyhow::bail!("слишком мало сообщений для консолидации");
    }

    let client_llm = LlmClient::from_config(&cfg)?;
    let emb_client = EmbeddingClient::from_config(&cfg.general.embedding.endpoint);
    let diary = Diary::new(
        &cfg.misc.diary_dir,
        emb_client,
        cfg.misc.diary_plagiarism_threshold,
        cfg.misc.diary_min_relatedness,
    );
    let mut diary_entries = diary.load_all().unwrap_or_else(|e| {
        eprintln!("[file-consolidate] дневник не загрузился: {}", e);
        Vec::new()
    });
    let prompts = Prompts::load_from_dir("prompts", &cfg)?;

    println!("[file-consolidate] вызываем consolidate_memory...");
    let rt = Runtime::new()?;
    let n = rt.block_on(furinya_diary::consolidate_memory(
        &client_llm,
        &diary,
        &prompts,
        &mut diary_entries,
        &cfg.misc.diary_dir,
        &conv,
        chat_id,
        cfg.misc.diary_keep_last,
        cfg.misc.active_top_k,
    ))?;
    println!("[file-consolidate] готово. Diary-записей сохранено: {}", n);

    // Показываем итоговый файл.
    let path = std::path::Path::new(&cfg.misc.diary_dir)
        .join("chat_history")
        .join(format!("{}.md", chat_id));
    println!("\n══════════ ИТОГ: {:?} ══════════", path);
    match std::fs::read_to_string(&path) {
        Ok(s) => {
            let max = 4000;
            if s.len() <= max {
                println!("{}", s);
            } else {
                // Обрезаем по ГРАНИЦЕ СИМВОЛА, а не по байту (UTF-8!).
                let cut = s
                    .char_indices()
                    .nth(max)
                    .map(|(i, _)| i)
                    .unwrap_or(s.len());
                println!("{}…\n[обрезано до {} симв из {}]", &s[..cut], max, s.len());
            }
        }
        Err(e) => println!("не удалось прочитать файл: {}", e),
    }
    Ok(())
}
