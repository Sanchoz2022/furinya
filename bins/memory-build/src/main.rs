//! memory-build — извлекает РЕАЛЬНЫЕ факты о юзере из per-чат выжимок
//! (data/diary/chat_history/*.md) в общую векторную память (data/diary/*.md).
//!
//! НЕ трогает TDLib. Только читает локальные выжимки и пишет diary через LLM.
//! Запуск: ./target/debug/memory-build
//! (опц.) CHAT_ID=5005580127 — можно ограничить, но по умолчанию берёт ВСЕ
//! выжимки из chat_history/.

use anyhow::Result;
use furinya_core::{config::Config, prompts::Prompts};
use furinya_diary::{Diary, EmbeddingClient};
use furinya_llm::LlmClient;
use tokio::runtime::Runtime;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()))
        .init();

    let cfg = Config::load("config.toml")?;
    let client_llm = LlmClient::from_config(&cfg)?;
    let emb_client = EmbeddingClient::from_config(&cfg.general.embedding.endpoint);
    let diary = Diary::new(
        &cfg.misc.diary_dir,
        emb_client,
        cfg.misc.diary_plagiarism_threshold,
        cfg.misc.diary_min_relatedness,
    );
    let mut diary_entries = diary.load_all().unwrap_or_else(|e| {
        eprintln!("[memory-build] дневник не загрузился: {}", e);
        Vec::new()
    });
    let prompts = Prompts::load_from_dir("prompts", &cfg)?;

    println!("[memory-build] извлекаем факты о юзере из chat_history/ ...");
    let rt = Runtime::new()?;
    let n = rt.block_on(furinya_diary::build_profile_memory(
        &client_llm,
        &diary,
        &prompts,
        &cfg.misc.diary_dir,
        &mut diary_entries,
        cfg.misc.diary_keep_last,
        cfg.misc.active_top_k,
    ))?;

    println!("[memory-build] готово. Извлечено фактов: {}", n);
    if n > 0 {
        println!("\n══════════ НОВЫЕ ЗАПИСИ В data/diary/ ═════════");
        // Показываем самые свежие записи (по убыванию id = ts).
        let mut recent: Vec<_> = diary_entries.iter().collect();
        recent.sort_by(|a, b| b.id.cmp(&a.id));
        for e in recent.iter().take(n) {
            println!("\n─── {} (conf={}) ───", e.id, e.metadata.confidence);
            println!("{}", e.body);
        }
    }
    Ok(())
}
