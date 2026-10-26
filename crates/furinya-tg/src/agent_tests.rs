//! Офлайн-интеграционный тест полного хода агента.
//!
//! Поднимается mock LLM-HTTP (SSE с tool_calls + JSON для оценки/заметки +
//! эмбеддинги) и тестовый `TdClient` без TDLib: команды уходят в канал, тест
//! их читает. Так проверяется весь wiring хода (оценка → инъекция → tool-loop →
//! память) без Telegram и без реальных API.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

use furinya_core::{Config, EndpointConfig, Prompts};
use furinya_diary::{Diary, EmbeddingClient};
use furinya_llm::{LlmClient, Message, SamplingParams};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::TdClient;
use crate::agent::handle_new_message;

/// Маркеры в промптах: mock различает служебные вызовы (оценка / ситуативная
/// заметка) по system-промпту — они неструмящиеся и без tools.
const ASSESS_MARK: &str = "ASSESS_MARK_7f3";
const SITUATION_MARK: &str = "SITUATION_MARK_7f3";
/// Маркер, по которому в system-промпте хода ищем инъекцию оценки.
const ASSESS_BLOCK_MARK: &str = "# Оценка хода (внутренняя)";

const CHAT_ID: i64 = 5005580127;
const MY_ID: i64 = 8337379413;
const SYSTEM_PROMPT: &str = "БАЗОВЫЙ СИСТЕМНЫЙ ПРОМПТ ХОДА";

// ─── mock LLM ────────────────────────────────────────────────────────────────

/// Очередь ответов основного хода: (имя инструмента, аргументы JSON).
type ToolReply = (String, String);

#[derive(Default)]
struct MockState {
    /// Ответы основного хода (stream + tools). Когда очередь пуста — stay_quiet.
    main_queue: Mutex<VecDeque<ToolReply>>,
    /// Все тела запросов (для проверок инъекции/блокировок).
    requests: Mutex<Vec<Value>>,
    /// Ответ на запрос оценки хода.
    assessment: Mutex<String>,
    /// Ответ на запрос ситуативной заметки.
    situation: Mutex<String>,
}

impl MockState {
    fn main_requests(&self) -> Vec<Value> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.get("tools").map(|t| !t.is_null()).unwrap_or(false))
            .cloned()
            .collect()
    }
}

/// Поднимает mock LLM на 127.0.0.1 и возвращает его base URL.
async fn start_mock(state: Arc<MockState>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((socket, _)) = listener.accept().await else {
                break;
            };
            let state = state.clone();
            tokio::spawn(async move {
                let _ = handle_conn(socket, state).await;
            });
        }
    });
    format!("http://{addr}/v1")
}

async fn handle_conn(mut socket: TcpStream, state: Arc<MockState>) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    while let Some((head, body)) = read_http(&mut socket, &mut buf).await? {
        let path = head
            .lines()
            .next()
            .unwrap_or("")
            .split_whitespace()
            .nth(1)
            .unwrap_or("/")
            .to_string();
        if path.ends_with("/embeddings") {
            let n = embedding_count(&body);
            let data: Vec<Value> = (0..n)
                .map(|i| json!({"embedding": [0.1, 0.2, 0.3], "index": i}))
                .collect();
            let resp = json!({"data": data}).to_string();
            write_http(&mut socket, "application/json", &resp).await?;
            continue;
        }
        let req: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
        state.requests.lock().unwrap().push(req.clone());
        let is_stream = req["stream"].as_bool().unwrap_or(false);
        let has_tools = req
            .get("tools")
            .map(|t| !t.is_null())
            .unwrap_or(false);
        if is_stream && has_tools {
            // Основной ход: отдаём следующий tool_call из очереди.
            let (name, args) = state
                .main_queue
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| ("stay_quiet".to_string(), "{}".to_string()));
            let chunk = json!({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_0",
                            "function": {"name": name, "arguments": args}
                        }]
                    }
                }]
            });
            let sse = format!("data: {chunk}\n\ndata: [DONE]\n\n");
            write_http(&mut socket, "text/event-stream", &sse).await?;
        } else {
            // Служебный вызов: оценка хода или ситуативная заметка.
            let sys = req["messages"]
                .as_array()
                .and_then(|m| m.first())
                .map(|m| m["content"].as_str().unwrap_or("").to_string())
                .unwrap_or_default();
            let content = if sys.contains(ASSESS_MARK) {
                state.assessment.lock().unwrap().clone()
            } else if sys.contains(SITUATION_MARK) {
                state.situation.lock().unwrap().clone()
            } else {
                "Ситуация: неизвестно\nОн: неизвестно\nЯ: неизвестно".to_string()
            };
            let resp = json!({
                "choices": [{
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop"
                }]
            })
            .to_string();
            write_http(&mut socket, "application/json", &resp).await?;
        }
    }
    Ok(())
}

/// Читает один HTTP-запрос из сокета. Возвращает (голова, тело).
async fn read_http(
    socket: &mut TcpStream,
    buf: &mut Vec<u8>,
) -> std::io::Result<Option<(String, Vec<u8>)>> {
    let head_end = loop {
        if let Some(pos) = find_subslice(buf, b"\r\n\r\n") {
            break pos + 4;
        }
        let mut tmp = [0u8; 4096];
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&tmp[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let len = head
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0);
    while buf.len() < head_end + len {
        let mut tmp = [0u8; 4096];
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = buf[head_end..head_end + len].to_vec();
    buf.drain(..head_end + len);
    Ok(Some((head, body)))
}

async fn write_http(socket: &mut TcpStream, content_type: &str, body: &str) -> std::io::Result<()> {
    let resp = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\n\
         connection: close\r\n\r\n{body}",
        body.as_bytes().len()
    );
    socket.write_all(resp.as_bytes()).await?;
    socket.flush().await
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

/// Сколько векторов вернуть: input — строка или массив строк.
fn embedding_count(body: &[u8]) -> usize {
    let req: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    match &req["input"] {
        Value::Array(a) => a.len().max(1),
        _ => 1,
    }
}

// ─── фикстура ────────────────────────────────────────────────────────────────

struct Fixture {
    dir: PathBuf,
    cfg: Config,
    llm: LlmClient,
    diary: Diary,
    prompts: Prompts,
    client: TdClient,
    rx: Receiver<Option<String>>,
    state: Arc<MockState>,
}

async fn fixture(tag: &str, main_queue: Vec<ToolReply>, assessment: &str) -> Fixture {
    let state = Arc::new(MockState::default());
    {
        let mut q = state.main_queue.lock().unwrap();
        for r in main_queue {
            q.push_back(r);
        }
        *state.assessment.lock().unwrap() = assessment.to_string();
        *state.situation.lock().unwrap() =
            "Ситуация: он написал, я ответила.\nОн: встревожен.\nЯ: собранна.".to_string();
    }
    let base_url = start_mock(state.clone()).await;

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = PathBuf::from("/tmp/opencode").join(format!(
        "furinya_agent_test_{tag}_{}_{}",
        std::process::id(),
        nanos
    ));
    let diary_dir = dir.join("diary");
    std::fs::create_dir_all(&diary_dir).unwrap();

    let toml = format!(
        r#"
[general]
papik_chat_id = {CHAT_ID}
telegram_my_id = {MY_ID}
character_name = "Furinya"
papik_name = "Александр"

[general.llm.endpoint]
baseUrl = "{base_url}"
model = "test-model"
bearerKey = "test-key"

[general.embedding.endpoint]
baseUrl = "{base_url}"
model = "test-embed"
bearerKey = "test-key"

[misc]
diary_dir = "{}"
max_message_parts = 3
llm_temperature = 0.85
situation_recent_n = 6
situation_top_k = 3
liveliness_typo_probability = 0.0
"#,
        diary_dir.display()
    );
    let cfg_path = dir.join("config.toml");
    std::fs::write(&cfg_path, toml).unwrap();
    let cfg = Config::load(&cfg_path).unwrap();

    let llm = LlmClient::new(
        &base_url,
        "test-key",
        "test-model",
        SamplingParams {
            temperature: 0.85,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            repetition_penalty: None,
        },
    );
    let diary = Diary::new(
        &diary_dir,
        EmbeddingClient::from_config(&EndpointConfig {
            base_url: base_url.clone(),
            bearer_key: "test-key".to_string(),
            model: "test-embed".to_string(),
            ..Default::default()
        }),
        0.97,
        0.55,
    );
    let mut extra: HashMap<String, String> = HashMap::new();
    extra.insert(
        "assess".to_string(),
        format!("{ASSESS_MARK}\nВерни три строки: СОСТОЯНИЕ ОН / СОСТОЯНИЕ Я / ПЛАН."),
    );
    extra.insert(
        "situation".to_string(),
        format!("{SITUATION_MARK}\nВерни заметку: Ситуация / Он / Я."),
    );
    let prompts = Prompts {
        system: "системный промпт".to_string(),
        character_base: String::new(),
        character_appearance: String::new(),
        extra,
    };
    let (client, rx) = TdClient::test_client();

    Fixture {
        dir,
        cfg,
        llm,
        diary,
        prompts,
        client,
        rx,
        state,
    }
}

/// Сообщение от Александра.
fn incoming(text: &str) -> Value {
    json!({
        "message": {
            "chat_id": CHAT_ID,
            "id": 42,
            "date": 1_700_000_000,
            "sender_id": {"user_id": CHAT_ID},
            "content": {
                "@type": "messageText",
                "text": {"@type": "formattedText", "text": text}
            }
        }
    })
}

/// Тексты сообщений, реально отправленных в чат (sendMessage).
fn sent_messages(rx: &Receiver<Option<String>>) -> Vec<String> {
    let mut out = Vec::new();
    while let Ok(Some(cmd)) = rx.try_recv() {
        let Ok(v) = serde_json::from_str::<Value>(&cmd) else {
            continue;
        };
        if v["@type"].as_str() == Some("sendMessage") {
            let text = v["input_message_content"]["text"]["text"]
                .as_str()
                .unwrap_or("")
                .to_string();
            out.push(text);
        }
    }
    out
}

fn send_tool(parts: &[&str]) -> ToolReply {
    let args = json!({"parts": parts}).to_string();
    ("send_telegram_message".to_string(), args)
}

// ─── тесты ───────────────────────────────────────────────────────────────────

#[tokio::test]
async fn live_turn_assesses_injects_and_sends_one_message() {
    let f = fixture(
        "live",
        vec![send_tool(&[
            "Ты написал, что тебе тяжело. Я не буду делать вид, что знаю, как это исправить. Расскажи, что именно случилось.",
        ])],
        "СОСТОЯНИЕ ОН: тревога, просит поддержки\nСОСТОЯНИЕ Я: спокойно, готова слушать\nПЛАН: признать боль и спросить, что нужно",
    )
    .await;

    let mut diary_entries = Vec::new();
    let mut tg_history = vec![Message::system("system")];
    let mut tg_history_full: Vec<(i64, bool, String, i64)> = Vec::new();

    handle_new_message(
        incoming("Саш, мне сегодня очень тяжело"),
        &f.llm,
        &f.client,
        &f.diary,
        &f.prompts,
        &mut diary_entries,
        &mut tg_history,
        &mut tg_history_full,
        SYSTEM_PROMPT,
        &f.cfg,
        MY_ID,
        false,
    )
    .await;

    // 1. Ровно одно сообщение ушло, текст — из tool_call.
    let sent = sent_messages(&f.rx);
    assert_eq!(sent.len(), 1, "ожидалось 1 сообщение, отправлено: {sent:?}");
    assert!(sent[0].contains("Я не буду делать вид"));

    // 2. Оценка инжектится в system-промпт хода.
    let main_reqs = f.state.main_requests();
    assert!(!main_reqs.is_empty(), "не было запросов основного хода");
    let sys = main_reqs[0]["messages"][0]["content"].as_str().unwrap_or("");
    assert!(
        sys.contains(ASSESS_BLOCK_MARK),
        "в system-промпт хода не попал блок оценки:\n{sys}"
    );
    assert!(sys.contains("СОСТОЯНИЕ ОН: тревога"));

    // 3. Per-chat состояние записано.
    let state_path = f.dir.join("diary").join("chat_state").join(format!("{CHAT_ID}.md"));
    let state_text = std::fs::read_to_string(&state_path).expect("chat_state не записан");
    assert!(state_text.contains("Он: тревога"), "{state_text}");

    // 4. Ситуативная заметка записана (атомарная, последовательная).
    let sit_path = f.dir.join("diary").join("situation").join(format!("{CHAT_ID}.md"));
    let sit_text = std::fs::read_to_string(&sit_path).expect("ситуативная заметка не записана");
    assert!(sit_text.contains("Ситуация:"), "{sit_text}");

    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn stay_quiet_on_live_message_is_blocked() {
    let f = fixture(
        "quiet_live",
        vec![
            ("stay_quiet".to_string(), "{}".to_string()),
            send_tool(&["Ты пишешь, я слышу. Что случилось?"]),
        ],
        "СОСТОЯНИЕ ОН: шарит\nСОСТОЯНИЕ Я: спокойно\nПЛАН: ответить",
    )
    .await;

    let mut diary_entries = Vec::new();
    let mut tg_history = vec![Message::system("system")];
    let mut tg_history_full: Vec<(i64, bool, String, i64)> = Vec::new();

    handle_new_message(
        incoming("Саш, привет"),
        &f.llm,
        &f.client,
        &f.diary,
        &f.prompts,
        &mut diary_entries,
        &mut tg_history,
        &mut tg_history_full,
        SYSTEM_PROMPT,
        &f.cfg,
        MY_ID,
        false,
    )
    .await;

    // Молчать на живое сообщение нельзя: модель получила запрет и ответила текстом.
    let sent = sent_messages(&f.rx);
    assert_eq!(sent.len(), 1, "ожидался 1 ответ, отправлено: {sent:?}");
    assert!(
        sent[0].contains("Ты пишешь"),
        "в чат ушло не то, что сказала модель: {sent:?}"
    );
    assert!(
        !sent[0].contains("зависла"),
        "generic-заглушка вместо ответа модели: {sent:?}"
    );

    // В следующем запросе есть tool-ответ с запретом молчать.
    let main_reqs = f.state.main_requests();
    assert!(
        main_reqs.len() >= 2,
        "ожидался повторный запрос основного хода, было {}",
        main_reqs.len()
    );
    let dump = main_reqs[1].to_string();
    assert!(dump.contains("ЗАПРЕЩЕНО"), "нет запрета молчать: {dump}");

    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn idle_stay_quiet_is_silent_and_writes_no_state() {
    let f = fixture(
        "idle",
        vec![("stay_quiet".to_string(), "{}".to_string())],
        "СОСТОЯНИЕ ОН: неизвестно\nСОСТОЯНИЕ Я: спокойно\nПЛАН: молчать",
    )
    .await;

    let mut diary_entries = Vec::new();
    let mut tg_history = vec![Message::system("system")];
    let mut tg_history_full: Vec<(i64, bool, String, i64)> = Vec::new();

    handle_new_message(
        incoming("тишина"),
        &f.llm,
        &f.client,
        &f.diary,
        &f.prompts,
        &mut diary_entries,
        &mut tg_history,
        &mut tg_history_full,
        SYSTEM_PROMPT,
        &f.cfg,
        MY_ID,
        true, // idle
    )
    .await;

    // Тишина: в чат ничего не ушло.
    let sent = sent_messages(&f.rx);
    assert!(sent.is_empty(), "idle-тишина не должна слать сообщения: {sent:?}");

    // Per-chat состояние на idle не пишется (его состояние неизвестно).
    let state_path = f.dir.join("diary").join("chat_state").join(format!("{CHAT_ID}.md"));
    assert!(
        !state_path.exists(),
        "на idle per-chat состояние писаться не должно"
    );

    // Но сама оценка на idle ЗАПУСКАЕТСЯ и её блок инжектится в system-промпт:
    // инициатива должна опираться на «СОСТОЯНИЕ Я / ПЛАН», а не на авось.
    let main_reqs = f.state.main_requests();
    assert!(!main_reqs.is_empty(), "не было запросов основного хода");
    let sys = main_reqs[0]["messages"][0]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        sys.contains(ASSESS_BLOCK_MARK),
        "на idle оценка не инжектится в system-промпт: {sys}"
    );

    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn parenthetical_action_is_glued_to_message_not_sent_alone() {
    let f = fixture(
        "action",
        vec![send_tool(&[
            "Ну и денёк сегодня выдался, если честно.",
            "(потягивается)",
        ])],
        "СОСТОЯНИЕ ОН: не писал — неизвестно\nСОСТОЯНИЕ Я: сонная, но довольная\nПЛАН: рассказать про свой день",
    )
    .await;

    let mut diary_entries = Vec::new();
    let mut tg_history = vec![Message::system("system")];
    let mut tg_history_full: Vec<(i64, bool, String, i64)> = Vec::new();

    handle_new_message(
        incoming("тишина"),
        &f.llm,
        &f.client,
        &f.diary,
        &f.prompts,
        &mut diary_entries,
        &mut tg_history,
        &mut tg_history_full,
        SYSTEM_PROMPT,
        &f.cfg,
        MY_ID,
        true,
    )
    .await;

    let sent = sent_messages(&f.rx);
    assert_eq!(
        sent.len(),
        1,
        "действие в скобках не должно уходить отдельным сообщением: {sent:?}"
    );
    assert!(
        sent[0].contains("(потягивается)"),
        "действие потерялось: {sent:?}"
    );

    let _ = std::fs::remove_dir_all(&f.dir);
}

#[tokio::test]
async fn broken_assessment_degrades_to_turn_without_state() {
    let f = fixture(
        "broken_assess",
        vec![send_tool(&[
            "Привет, Саш. Я как раз думала о вчерашнем разговоре — ты дочитал ту книгу?",
        ])],
        "какой-то мусор без нужных меток",
    )
    .await;

    let mut diary_entries = Vec::new();
    let mut tg_history = vec![Message::system("system")];
    let mut tg_history_full: Vec<(i64, bool, String, i64)> = Vec::new();

    handle_new_message(
        incoming("Саш, привет"),
        &f.llm,
        &f.client,
        &f.diary,
        &f.prompts,
        &mut diary_entries,
        &mut tg_history,
        &mut tg_history_full,
        SYSTEM_PROMPT,
        &f.cfg,
        MY_ID,
        false,
    )
    .await;

    // Ход отработал: ответ ушёл.
    let sent = sent_messages(&f.rx);
    assert_eq!(sent.len(), 1, "ожидался 1 ответ, отправлено: {sent:?}");

    // Оценка не разобралась → в system-промпт хода блока оценки нет.
    let main_reqs = f.state.main_requests();
    let sys = main_reqs[0]["messages"][0]["content"].as_str().unwrap_or("");
    assert!(
        !sys.contains(ASSESS_BLOCK_MARK),
        "при битой оценке блок оценки инжектиться не должен:\n{sys}"
    );

    // Per-chat состояние не записано.
    let state_path = f.dir.join("diary").join("chat_state").join(format!("{CHAT_ID}.md"));
    assert!(
        !state_path.exists(),
        "при битой оценке chat_state писаться не должен"
    );

    let _ = std::fs::remove_dir_all(&f.dir);
}
