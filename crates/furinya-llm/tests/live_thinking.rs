//! Live-тесты против облачного Mistral API (`mistral-large-2512`).
//!
//! Ключ берётсякак обычно: из config.toml → general.llm.endpoint.bearerKey,
//! либо из env `MISTRAL_API_KEY` (перекрывает пустой конфиг). Нет ключа —
//! тест тихо пропускается.
//!
//! Запуск: `cargo test -p furinya-llm --test live_thinking -- --ignored --nocapture`
//! (игнорируется по умолчанию, чтобы не ломать CI без ключа).

use furinya_core::config::Config;
use furinya_llm::{LlmClient, Message, build_tools};

/// Путь к config.toml из корня воркспейса (.cwd у теста = crates/furinya-llm).
fn load_config() -> Option<Config> {
    let root = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .ancestors()
        .nth(2)
        .unwrap()
        .join("config.toml");
    Config::load(&root).ok()
}

/// Есть ли ключ в конфиге или в окружении.
fn have_key(cfg: &Config) -> bool {
    let env = std::env::var("MISTRAL_API_KEY")
        .ok()
        .map(|k| !k.trim().is_empty())
        .unwrap_or(false);
    !cfg.general.llm.endpoint.bearer_key.trim().is_empty() || env
}

#[tokio::test]
#[ignore]
async fn live_stream_replies_content() {
    let Some(cfg) = load_config() else {
        eprintln!("пропущен: config.toml не найден");
        return;
    };
    if !have_key(&cfg) {
        eprintln!("пропущен: нет ключа (bearerKey в config.toml или MISTRAL_API_KEY в env)");
        return;
    }
    let client = LlmClient::from_config(&cfg).expect("client build");

    let messages = vec![Message::user(
        "Привет! Напиши одну короткую фразу о том, о чём ты сейчас думаешь.",
    )];

    let mut content = String::new();
    let mut reasoning = String::new();
    client
        .chat_stream(
            messages,
            None,
            |d| content.push_str(d),
            |r| reasoning.push_str(r),
        )
        .await
        .expect("stream должен завершиться без ошибки");

    println!("--- CONTENT ---\n{content}");
    if !reasoning.trim().is_empty() {
        println!("--- REASONING (bonus) ---\n{reasoning}");
    }

    assert!(
        !content.trim().is_empty(),
        "content пустой — модель ничего не ответила"
    );
}

#[tokio::test]
#[ignore]
async fn live_stream_native_tool_call() {
    let Some(cfg) = load_config() else {
        eprintln!("пропущен: config.toml не найден");
        return;
    };
    if !have_key(&cfg) {
        eprintln!("пропущен: нет ключа (bearerKey в config.toml или MISTRAL_API_KEY в env)");
        return;
    }
    let client = LlmClient::from_config(&cfg).expect("client build");

    // Просим модель поставить реакцию — должна вызвать react_with_emoji.
    let messages = vec![
        Message::system(
            "Ты агент. Если сообщение пользователя — просто приветствие, ставь emoji-реакцию через react_with_emoji, а не отвечай текстом.",
        ),
        Message::user("Привет!"),
    ];

    let tools = build_tools();
    let tool_calls = client
        .chat_stream(messages, Some(tools), |_d| {}, |_r| {})
        .await
        .expect("stream должен завершиться без ошибки");

    println!("--- TOOL CALLS ---");
    for tc in &tool_calls {
        println!("name={} args={}", tc.function.name, tc.function.arguments);
    }

    assert!(
        !tool_calls.is_empty(),
        "модель не вернула native tool_call — tool calling не работает!"
    );
    assert!(
        tool_calls
            .iter()
            .any(|tc| tc.function.name == "react_with_emoji"
                || tc.function.name == "send_telegram_message"),
        "ожидался react_with_emoji или send_telegram_message, получено: {:?}",
        tool_calls.iter().map(|t| &t.function.name).collect::<Vec<_>>()
    );
}
