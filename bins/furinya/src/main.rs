//! Furinya — точка входа.
//!
//! Загружает конфиг/промпты/дневник, выбирает режим (консоль или Telegram).
//! Весь функционал общения живёт в `furinya_tg`, память — в `furinya_diary`,
//! LLM — в `furinya_llm`. main.rs только связывает их.

use std::io::{self, Write};

use anyhow::Result;
use furinya_core::config::Config;
use furinya_core::logging::{estimate_tokens, init_logging};
use furinya_core::prompts::Prompts;
use furinya_diary::{Diary, EmbeddingClient};
use furinya_llm::{
    LlmClient, Message, extract_clean_reply, extract_thinking, extract_thinking_from_content,
    parse_tool_calls,
};
use tracing::{error, info, warn};

/// Максимальное число итераций tool-loop в консольном режиме.
const MAX_TOOL_ITERS: u32 = 100;

/// Показывать ли раздумья модели в терминале (FURINYA_SHOW_THINKING=1).
fn show_thinking_enabled() -> bool {
    std::env::var("FURINYA_SHOW_THINKING")
        .map(|v| v == "1")
        .unwrap_or(false)
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = init_logging("furinya.log");
    info!("[furinya] старт ядра (интерактивный чат)");

    let cfg = Config::load("config.toml")?;
    info!(
        "[furinya] конфиг: персонаж='{}', llm='{}'",
        cfg.general.character_name, cfg.general.llm.endpoint.base_url
    );
    // Резолвнутые возможности — чтобы TOML-ошибка раздела (dotted-ключи после
    // [[...loras]] и т.п.) не выключала голос/слух молча: видно в логе на старте.
    info!(
        "[furinya] возможности: фото={} ({}), голос={} ({}), слух={}, веб={}",
        cfg.capabilities.take_photo.enabled,
        cfg.capabilities.take_photo.backend,
        cfg.capabilities.record_voice.enabled,
        cfg.capabilities.record_voice.backend,
        cfg.capabilities.hearing.enabled,
        cfg.capabilities.web_search.enabled,
    );
    if cfg.capabilities.record_voice.enabled {
        let fish_key = std::env::var("FISH_API_KEY")
            .map(|k| !k.trim().is_empty())
            .unwrap_or(false)
            || !cfg.capabilities.record_voice.fish.key.trim().is_empty();
        info!(
            "[furinya] голос: fish referenceId={} ключ={}",
            if cfg.capabilities.record_voice.fish.reference_id.trim().is_empty() {
                "нет"
            } else {
                "есть"
            },
            if fish_key { "есть" } else { "НЕТ" },
        );
    }

    let prompts = Prompts::load_from_dir("prompts", &cfg)?;
    // Режим Telegram выбирается ниже (telegram_enabled + нет FURINYA_CONSOLE).
    let telegram_mode = cfg.general.telegram_enabled && std::env::var("FURINYA_CONSOLE").is_err();
    let full_env = std::env::var("FURINYA_FULL_PROMPT")
        .map(|v| v == "1")
        .unwrap_or(false);
    let raw_mode = std::env::var("FURINYA_RAW")
        .map(|v| v == "1")
        .unwrap_or(false);
    // В Telegram нужен ПОЛНЫЙ system.md: оценка хода, ответ только через
    // send_telegram_message, анти-повтор, правила idle/сна. Раньше прод брал
    // build_character_prompt() (только личность) — модель не знала ни про
    // оценку, ни про инструменты, и скатывалась в дежурные формулы.
    // В консоли tool-loop нет, там достаточно личности (если не задан FULL).
    let use_full = telegram_mode || full_env;
    let mut system_prompt = if use_full {
        prompts.build_system_prompt()
    } else {
        prompts.build_character_prompt()
    };
    if full_env {
        system_prompt.push_str(
            "\n\n# ОТЛАДКА (терминал)\nНа КАЖДОЕ сообщение пользователя ПЕРВЫМ делом вызови инструмент #ask для освежения контекста. ПЕРЕД ответом ОБЯЗАТЕЛЬНО напиши служебные строки:\n- thought: <рассуждения>\n- emotion: <эмоция>\n- intention: <намерение>\nТолько ПОСЛЕ них пиши ответ.",
        );
    }

    println!(
        "[furinya] промпты загружены: system={} симв, character_base={} симв, appearance={} симв{}",
        prompts.system.len(),
        prompts.character_base.len(),
        prompts.character_appearance.len(),
        if full_env {
            " (FULL + thinking)"
        } else if use_full {
            " (FULL)"
        } else {
            ""
        }
    );

    let client = LlmClient::from_config(&cfg)?;
    let mut history: Vec<Message> = vec![Message::system(system_prompt.clone())];

    let emb_client = EmbeddingClient::from_config(&cfg.general.embedding.endpoint);
    let embed_identity = emb_client.identity();
    let diary = Diary::new(
        &cfg.misc.diary_dir,
        emb_client,
        cfg.misc.diary_plagiarism_threshold,
        cfg.misc.diary_min_relatedness,
    );
    let mut diary_entries = diary.load_all().unwrap_or_else(|e| {
        eprintln!("[furinya] не удалось загрузить дневник: {}", e);
        Vec::new()
    });
    println!(
        "[furinya] дневник: загружено {} записей (embedding={}, plagiarism>={:.2}, relatedness>={:.2})",
        diary_entries.len(),
        cfg.general.embedding.endpoint.base_url,
        cfg.misc.diary_plagiarism_threshold,
        cfg.misc.diary_min_relatedness,
    );

    migrate_embeddings_if_needed(&cfg, &diary, &mut diary_entries, &embed_identity).await;

    if telegram_mode {
        info!("[furinya] telegram_enabled=true — запуск TDLib режима");
        if std::env::var("FURINYA_TD_QUIET").is_ok() {
            use std::os::unix::io::IntoRawFd;
            if let Ok(devnull) = std::fs::OpenOptions::new().write(true).open("/dev/null") {
                let fd = devnull.into_raw_fd();
                unsafe { libc::dup2(fd, 2) };
            }
        }
        return furinya_tg::agent::run_telegram_mode(
            &cfg,
            &client,
            &diary,
            &prompts,
            &mut diary_entries,
            &system_prompt,
        )
        .await;
    }

    println!("────────────────────────────────────────────");
    println!(
        " Чат с {} (для выхода: пустая строка / exit)",
        cfg.general.character_name
    );
    println!("────────────────────────────────────────────");

    let mut input = String::new();
    loop {
        print!("\nты> ");
        io::stdout().flush().ok();
        input.clear();
        if io::stdin().read_line(&mut input)? == 0 {
            println!("\n[furinya] выход");
            break;
        }
        let text = input.trim();
        if text.eq_ignore_ascii_case("exit") || text.eq_ignore_ascii_case("выход") {
            println!("[furinya] выход");
            break;
        }
        if text.is_empty() {
            println!("[furinya] консолидация памяти (Enter)...");
            let chat_turns = furinya_diary::ChatTurn::from_messages_no_dates(
                &history,
                &cfg.general.character_name,
                &prompts.papik_name(),
            );
            let n = furinya_diary::consolidate_memory(
                &client,
                &diary,
                &prompts,
                &mut diary_entries,
                &cfg.misc.diary_dir,
                &chat_turns,
                cfg.general.papik_chat_id,
                cfg.misc.diary_keep_last,
                cfg.misc.active_top_k,
            )
            .await?;
            println!("[furinya] консолидировано {} записей", n);
            println!("[furinya] выход");
            break;
        }

        if let Some(rest) = text.strip_prefix("!memory") {
            let tags: Vec<String> = rest
                .split_whitespace()
                .filter(|t| t.starts_with('#'))
                .map(|t| t.trim_start_matches('#').to_string())
                .collect();
            print_memory(&diary_entries, &tags);
            continue;
        }

        if text.trim() == "!consolidate" {
            let chat_turns = furinya_diary::ChatTurn::from_messages_no_dates(
                &history,
                &cfg.general.character_name,
                &prompts.papik_name(),
            );
            let n = furinya_diary::consolidate_memory(
                &client,
                &diary,
                &prompts,
                &mut diary_entries,
                &cfg.misc.diary_dir,
                &chat_turns,
                cfg.general.papik_chat_id,
                cfg.misc.diary_keep_last,
                cfg.misc.active_top_k,
            )
            .await?;
            println!("[furinya] консолидировано {} записей", n);
            diary_entries = diary.load_all()?;
            continue;
        }

        let history_text: String = history
            .iter()
            .map(|m| match &m.content {
                furinya_llm::MessageContent::Text(s) => s.clone(),
                furinya_llm::MessageContent::Parts(_) => String::new(),
            })
            .collect::<Vec<_>>()
            .join("\n");
        let est_tokens = estimate_tokens(&history_text);
        if est_tokens as u32 > cfg.misc.context_token_limit {
            warn!(
                "CONTEXT OVERFLOW: оценка {est_tokens} токенов > лимит {} — укорачиваю историю",
                cfg.misc.context_token_limit
            );
            let system_msg = history.first().cloned();
            if history.len() > 9 {
                history = history.split_off(history.len().saturating_sub(8));
                if let Some(sys) = system_msg {
                    history.insert(0, sys);
                }
            }
        }

        history.push(Message::user(text.to_string()));

        if std::env::var("FURINYA_DIARY_TEST")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            match diary.embedding().embed(text).await {
                Ok(vec) => {
                    // Тот же центрированный косинус, что и в рабочем RAG:
                    // у mistral-embed сырое сходство ~0.9 даже у мусора.
                    let valid: Vec<_> = diary_entries
                        .iter()
                        .filter(|e| !e.metadata.embedding.is_empty())
                        .collect();
                    let mean = if valid.len() >= furinya_diary::CENTER_MIN_ENTRIES {
                        furinya_diary::corpus_mean(
                            valid.iter().map(|e| e.metadata.embedding.as_slice()),
                        )
                    } else {
                        Vec::new()
                    };
                    let mut all: Vec<_> = valid
                        .into_iter()
                        .map(|e| {
                            (
                                furinya_diary::centered_cosine(
                                    &vec,
                                    &e.metadata.embedding,
                                    &mean,
                                ),
                                e,
                            )
                        })
                        .collect();
                    all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    let hits: Vec<_> = all
                        .iter()
                        .filter(|(s, _)| *s >= cfg.misc.diary_min_relatedness)
                        .collect();
                    println!(
                        "🔎 RAG: найдено {} релевантных (порог {:.2}), топ-3:",
                        hits.len(),
                        cfg.misc.diary_min_relatedness
                    );
                    for (sim, e) in all.iter().take(3) {
                        let preview: String = e.body.lines().take(2).collect::<Vec<_>>().join(" ");
                        println!("  [{:.3}] {}", sim, preview);
                    }
                    if hits.is_empty() && all.is_empty() {
                        println!("  (пусто — дневник молчит или нет эмбеддингов)");
                    }
                }
                Err(e) => eprintln!("[furinya] RAG ошибка эмбеддинга: {}", e),
            }
        }

        let mut tool_iter = 0;
        loop {
            print!("{}> ", cfg.general.character_name);
            io::stdout().flush().ok();
            let mut full = String::new();
            let mut reasoning_acc = String::new();
            let stream_res = client
                .chat_stream(
                    history.clone(),
                    None,
                    |delta: &str| {
                        print!("{delta}");
                        io::stdout().flush().ok();
                        full.push_str(delta);
                    },
                    |reasoning: &str| {
                        if show_thinking_enabled() {
                            print!("\x1b[33m{reasoning}");
                            io::stdout().flush().ok();
                        }
                        reasoning_acc.push_str(reasoning);
                    },
                )
                .await
                .map(drop);

            match stream_res {
                Ok(()) => {
                    println!();
                    let calls = parse_tool_calls(&full);
                    if calls.is_empty() {
                        if show_thinking_enabled() {
                            let thinking = extract_thinking(&reasoning_acc)
                                .or_else(|| extract_thinking_from_content(&full));
                            if let Some(thinking) = thinking {
                                println!("\x1b[33m💭 размышления:\x1b[0m");
                                println!("{thinking}");
                                tracing::debug!(thinking = %thinking, "model thinking chain");
                            }
                        }
                        if !raw_mode {
                            match extract_clean_reply(&full) {
                                Some(clean) => {
                                    println!("──────────── ответ ────────────");
                                    println!("{clean}");
                                    println!("────────────────────────────────");
                                    history.push(Message::assistant(clean));
                                }
                                None => {
                                    println!("──────────── ответ ────────────");
                                    println!("(модель не сгенерировала текст — только служебные поля)");
                                    println!("────────────────────────────────");
                                    history.push(Message::assistant(full.trim().to_string()));
                                }
                            }
                            break;
                        } else {
                            if show_thinking_enabled() {
                                let thinking = extract_thinking(&reasoning_acc)
                                    .or_else(|| extract_thinking_from_content(&full));
                                if let Some(thinking) = thinking {
                                    println!("💭 размышления (из raw):");
                                    println!("{thinking}");
                                    tracing::debug!(thinking = %thinking, "model thinking chain (raw)");
                                }
                            }
                            history.push(Message::assistant(full.trim().to_string()));
                            break;
                        }
                    }

                    if show_thinking_enabled() {
                        let thinking = extract_thinking(&reasoning_acc)
                            .or_else(|| extract_thinking_from_content(&full));
                        if let Some(thinking) = thinking {
                            println!("💭 размышления:");
                            println!("{thinking}");
                            tracing::debug!(thinking = %thinking, "model thinking chain (tool)");
                        }
                    }
                    tool_iter += 1;
                    if tool_iter > MAX_TOOL_ITERS {
                        warn!("[furinya] превышен лимит tool-итераций, прерываю");
                        break;
                    }
                    for call in &calls {
                        println!("  ⟳ вызов инструмента: {}", call.name);
                        if show_thinking_enabled() {
                            println!("    аргументы: {}", call.args_json);
                        }
                        let result = execute_tool(
                            &call.name,
                            &call.args_json,
                            &diary,
                            &mut diary_entries,
                            &cfg.misc.diary_dir,
                            &cfg,
                        )
                        .await;
                        history.push(Message::tool(result));
                    }
                    history.push(Message::assistant(full.trim().to_string()));
                    continue;
                }
                Err(e) => {
                    error!("[furinya] ОШИБКА LLM: {e}");
                    error!(
                        "[furinya] Проверь, что инференс поднят на {}",
                        cfg.general.llm.endpoint.base_url
                    );
                    history.pop();
                    break;
                }
            }
        }
    }

    info!("[furinya] ядро остановлено.");
    Ok(())
}

fn extract_query_arg(args: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args).ok()?;
    if let Some(q) = v.get("query").and_then(|q| q.as_str()) {
        return Some(q.to_string());
    }
    if let Some(s) = v
        .get("entry_details")
        .and_then(|e| e.get("query"))
        .and_then(|q| q.as_str())
    {
        return Some(s.to_string());
    }
    None
}

async fn execute_tool(
    name: &str,
    args: &str,
    diary: &Diary,
    diary_entries: &mut Vec<furinya_diary::Entry>,
    _diary_dir: &str,
    cfg: &Config,
) -> String {
    match name {
        "ask" => {
            let query_text = extract_query_arg(args).unwrap_or_default();
            if query_text.is_empty() {
                return "Дневник: пустой запрос #ask.".to_string();
            }
            match diary.embedding().embed(&query_text).await {
                Ok(vec) => {
                    let hits = diary.query(diary_entries, &vec);
                    if hits.is_empty() {
                        "Дневник: релевантных воспоминаний не найдено.".to_string()
                    } else {
                        let mut out = String::from("Дневник: найдено воспоминаний:\n");
                        for h in &hits {
                            out.push_str(&format!(
                                "[{:.2}] {}\n",
                                h.relatedness,
                                h.entry.body.lines().next().unwrap_or(&h.entry.body)
                            ));
                            out.push_str(&h.entry.body);
                            out.push_str("\n---\n");
                        }
                        out
                    }
                }
                Err(e) => format!("Дневник: ошибка эмбеддинга запроса: {e}"),
            }
        }
        "diary_page" => match diary.save_entries_from_text(args, diary_entries, 20, cfg.misc.diary_keep_last, cfg.misc.active_top_k).await {
            Ok(saved) => {
                let n = saved.len();
                diary_entries.extend(saved);
                format!("Дневник: сохранено {n} новых записей.")
            }
            Err(e) => format!("Дневник: ошибка сохранения: {e}"),
        },
        other => format!("Инструмент '{other}' пока не реализован."),
    }
}

fn print_memory(entries: &[furinya_diary::Entry], tags: &[String]) {
    let filtered: Vec<&furinya_diary::Entry> = if tags.is_empty() {
        entries.iter().collect()
    } else {
        entries
            .iter()
            .filter(|e| e.metadata.tags.iter().any(|t| tags.contains(t)))
            .collect()
    };

    let title = if tags.is_empty() {
        format!("ПАМЯТЬ (ДНЕВНИК) — {} записей", entries.len())
    } else {
        format!(
            "ПАМЯТЬ (теги: {}) — {} из {} записей",
            tags.join(", #"),
            filtered.len(),
            entries.len()
        )
    };

    println!("\n═════════ {} ═════════", title);
    if filtered.is_empty() {
        println!("  (пусто)");
        return;
    }
    let mut sorted: Vec<&&furinya_diary::Entry> = filtered.iter().collect();
    sorted.sort_by(|a, b| a.id.cmp(&b.id));
    for e in sorted {
        let tags_str = if e.metadata.tags.is_empty() {
            String::new()
        } else {
            format!(" [теги: {}]", e.metadata.tags.join(", "))
        };
        println!(
            "\n──────────── #{} (score={:.2}, used={}, conf={:.2}){} ────────────",
            e.id, e.metadata.score, e.metadata.usage_count, e.metadata.confidence, tags_str
        );
        println!("{}", e.body);
    }
    println!("\n════════════════════════════════════════════════════════");
}

/// Пересчёт векторной памяти при смене модели эмбеддингов.
///
/// Старые векторы (напр. Mistral) лежат в другом пространстве, чем новые
/// (локальный bge-m3), даже при совпадении размерности — cosine даёт мусор.
/// Сравниваем идентификатор эмбеддера (`baseUrl|model`) с маркером
/// `data/diary/.embed_model`; при расхождении пересчитываем дневник и
/// ситуационный индекс папика и только затем пишем маркер. Если эмбеддер
/// недоступен — маркер не пишем, пересчёт повторится на следующем старте.
async fn migrate_embeddings_if_needed(
    cfg: &Config,
    diary: &Diary,
    entries: &mut [furinya_diary::Entry],
    identity: &str,
) {
    let marker = std::path::Path::new(&cfg.misc.diary_dir).join(".embed_model");
    let prev = std::fs::read_to_string(&marker)
        .ok()
        .map(|s| s.trim().to_string());
    if prev.as_deref() == Some(identity) {
        return;
    }
    info!(
        "[furinya] смена эмбеддера: {} -> {} — пересчитываю память",
        prev.as_deref().unwrap_or("<нет>"),
        identity
    );
    match diary.reembed_all(entries).await {
        Ok(n) => {
            info!("[furinya] дневник: пересчитано {} эмбеддингов", n);
            let sit =
                furinya_diary::SituationStore::new(&cfg.misc.diary_dir, cfg.general.papik_chat_id);
            match sit.reindex(diary.embedding()).await {
                Ok(m) => info!("[furinya] ситуация: пересчитано {} эмбеддингов", m),
                Err(e) => warn!("[furinya] ситуация: пересчёт не удался: {e:#}"),
            }
            let _ = std::fs::write(&marker, identity);
        }
        Err(e) => warn!(
            "[furinya] пересчёт дневника не удался ({e:#}) — маркер не пишу, повтор на следующем старте"
        ),
    }
}
