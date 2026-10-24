//! furinya-tg — нативный Telegram user-account клиент на базе TDLib (tdjson).
//!
//! Использует JSON FFI интерфейс `libtdjson.so` (как в kuni), БЕЗ сборки tdlib
//! из исходников. Грузим .so через `libloading`.
//!
//! Архитектура потокобезопасности:
//! - ОДИН worker-поток владеет `*mut c_void` (handle клиента TDLib).
//! - Все `send()` и `receive()` выполняются ТОЛЬКО в этом потоке (TDLib требует,
//!   чтобы `receive` звался из одного потока).
//! - `TdClient::send()` кладёт команду в `Sender<String>`; worker выполняет её.
//! - Worker кладёт полученные update-ы в `Receiver<Value>` (для `run_telegram_mode`).
//! - При `Drop` TdClient шлёт спец-команду `None` в канал → worker вызывает
//!   `td_json_client_destroy` и завершается. Никакого double-free.

pub mod assessment;
pub mod auth;
pub mod agent;
pub mod idle;

#[cfg(test)]
mod agent_tests;

use libloading::{Library, Symbol};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use rand::Rng;
use tokio::sync::oneshot;
use tracing::info;

/// Слегка «портит» текст для живости: одна случайная опечатка в одном слове
/// (транспозиция / удвоение / потеря последней буквы). Возвращает None, если
/// портить нечего (короткий текст, нет слов >=5 букв).
pub fn corrupt_text_once(text: &str) -> Option<String> {
    // Находим "слова" (runы букв) с границами байт.
    let mut words: Vec<(usize, usize)> = Vec::new();
    let mut cur_start: Option<usize> = None;
    let mut cur_end = 0usize;
    for (i, c) in text.char_indices() {
        if c.is_alphabetic() {
            if cur_start.is_none() {
                cur_start = Some(i);
            }
            cur_end = i + c.len_utf8();
        } else if let Some(s) = cur_start.take() {
            words.push((s, cur_end));
        }
    }
    if let Some(s) = cur_start.take() {
        words.push((s, cur_end));
    }
    // Годятся только слова >=5 букв (мало повредить смысл).
    let candidates: Vec<(usize, usize)> = words
        .into_iter()
        .filter(|(s, e)| text[*s..*e].chars().count() >= 5)
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let mut rng = rand::thread_rng();
    let (s, e) = candidates[rng.gen_range(0..candidates.len())];
    let word_chars: Vec<char> = text[s..e].chars().collect();
    let kind: u8 = rng.gen_range(0..3);
    let mutated: String = match kind {
        0 => {
            // Транспозиция двух соседних букв в середине слова.
            let mid = rng.gen_range(1..word_chars.len() - 1);
            let mut v = word_chars.clone();
            v.swap(mid, mid + 1);
            v.into_iter().collect()
        }
        1 => {
            // Удвоение случайной буквы.
            let mid = rng.gen_range(0..word_chars.len());
            let mut v = word_chars.clone();
            v.insert(mid, v[mid]);
            v.into_iter().collect()
        }
        _ => {
            // Потеря последней буквы.
            word_chars[..word_chars.len() - 1].iter().collect()
        }
    };
    Some(format!("{}{}{}", &text[..s], mutated, &text[e..]))
}

type TdCreate = unsafe extern "C" fn() -> *mut std::ffi::c_void;
type TdSend = unsafe extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_char);
type TdReceive = unsafe extern "C" fn(*mut std::ffi::c_void, f64) -> *mut std::ffi::c_char;
type TdExecute =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_char) -> *mut std::ffi::c_char;
type TdDestroy = unsafe extern "C" fn(*mut std::ffi::c_void);

/// Безопасная обёртка над TDLib JSON-клиентом.
///
/// Владеет каналом команд к worker-потоку. Сам handle НЕ хранит (он в потоке).
/// Клон — дешёвый (канал команд + Arc-состояния), так что клиенты поллеров
/// новостей и файловых загрузок ходят через свои клоны.
#[derive(Clone)]
pub struct TdClient {
    /// Команды worker-потоку. `None` = сигнал завершения (destroy + exit).
    cmd_tx: Sender<Option<String>>,
    /// Pending-загрузки файлов: file_id → oneshot-отправитель локального пути.
    /// Заполняется receive-циклом при `updateFile` (local.path заполнен).
    downloads: Arc<Mutex<HashMap<i64, oneshot::Sender<String>>>>,
    /// Pending-запросы (getFile и др.): extra_id → oneshot-отправитель ответа.
    pending_queries: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>,
    /// Очередь запланированных «правок опечаток» (живость): (chat_id,
    /// правильный текст). При updateMessageSendSucceeded worker берёт голову
    /// очереди и через 1.4–2.8с редактирует сообщение на верный текст.
    pending_edits: Arc<Mutex<VecDeque<(i64, String)>>>,
}

impl TdClient {
    /// Запрашивает скачивание файла из TDLib и ДОЖИДАЕТСЯ его завершения
    /// (через `updateFile` с заполненным `local.path`). Возвращает локальный
    /// путь к скачанному файлу. Используется для фото/голосовых юзера.
    ///
    /// `priority` — 1..32 (16 = обычный). `offset`/`limit` = 0 = весь файл.
    /// Таймаут 300с — фото с телефона (50MP) и длинные голосовые могут качаться долго.
    pub async fn download_file(
        &self,
        file_id: i64,
        priority: i32,
    ) -> anyhow::Result<String> {
        let (tx, rx) = oneshot::channel::<String>();
        {
            let mut pending = self.downloads.lock().unwrap();
            pending.insert(file_id, tx);
        }
        self.send(&serde_json::json!({
            "@type": "downloadFile",
            "file_id": file_id,
            "priority": priority,
            "offset": 0,
            "limit": 0,
            "synchronous": false
        }));
        // TDLib может УЖЕ держать файл в кэше (is_downloading_completed=true),
        // и updateFile отлетит ДО того, как receive-цикл впишет file_id в pending
        // (race condition). Поэтому сразу после downloadFile делаем getFile и,
        // если файл уже готов с путём — отдаём путь, не дожидаясь updateFile.
        if let Ok(Some(path)) = self
            .get_file_path(file_id)
            .await
        {
            info!("[tg][debug] download_file: get_file_path вернул путь сразу: {}", path);
            let _ = self.downloads.lock().unwrap().remove(&file_id);
            return Ok(path);
        } else {
            info!("[tg][debug] download_file: get_file_path не дал путь, жду updateFile (file_id={})", file_id);
        }
        // Ждём локальный путь от receive-цикла (или таймаут).
        // Фото с телефона (50MP) и длинные голосовые могут качаться долго,
        // поэтому таймаут щедрый — 300с (не 60, иначе большие файлы падают).
        match tokio::time::timeout(Duration::from_secs(300), rx).await {
            Ok(Ok(path)) if !path.is_empty() => Ok(path),
            Ok(Ok(_)) => anyhow::bail!("downloadFile: пустой путь для file_id={}", file_id),
            Ok(Err(_)) => anyhow::bail!("downloadFile: канал закрыт до updateFile (file_id={})", file_id),
            Err(_) => {
                // Таймаут — убираем orphan-ожидание.
                self.downloads.lock().unwrap().remove(&file_id);
                anyhow::bail!("downloadFile: таймаут 300с (file_id={})", file_id)
            }
        }
    }
}

/// Дефолтный путь к libtdjson (рядом с бинарником в dev-режиме).
pub fn default_lib_path() -> PathBuf {
    PathBuf::from("libs/libtdjson.so")
}

#[cfg(test)]
impl TdClient {
    /// Тестовый клиент БЕЗ TDLib: команды уходят в канал, который читает тест.
    /// Нужен, чтобы прогнать полный ход агента офлайн (без Telegram).
    pub fn test_client() -> (Self, Receiver<Option<String>>) {
        let (cmd_tx, rx) = channel();
        (
            Self {
                cmd_tx,
                downloads: Arc::new(Mutex::new(HashMap::new())),
                pending_queries: Arc::new(Mutex::new(HashMap::new())),
                pending_edits: Arc::new(Mutex::new(VecDeque::new())),
            },
            rx,
        )
    }
}

impl TdClient {
    /// Отправляет JSON-команду в TDLib (через worker-поток).
    pub fn send(&self, json: &Value) {
        let s = json.to_string();
        let _ = self.cmd_tx.send(Some(s));
    }

    /// Запрашивает информацию о файле через `getFile` и возвращает его локальный
    /// путь (если `is_downloading_completed` и путь не пустой). Используется
    /// `download_file` для обхода race condition: TDLib может держать файл в
    /// кэше, и `updateFile` отлетает ДО регистрации ожидания — тогда `getFile`
    /// даёт путь сразу, без ожидания `updateFile`.
    ///
    /// Возвращает `Ok(Some(path))` если файл уже готов, `Ok(None)` если ещё не
    /// скачан (нужно дождаться `updateFile`), `Err` при сбое запроса.
    pub async fn get_file_path(&self, file_id: i64) -> anyhow::Result<Option<String>> {
        let extra = format!("getfile_{}_{}", file_id, std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0));
        info!("[tg][debug] get_file_path: запрашиваю getFile file_id={} extra={}", file_id, extra);
        let (tx, rx) = oneshot::channel::<Value>();
        {
            let mut pq = self.pending_queries.lock().unwrap();
            pq.insert(extra.clone(), tx);
        }
        self.send(&serde_json::json!({
            "@type": "getFile",
            "file_id": file_id,
            "@extra": extra
        }));
        match tokio::time::timeout(Duration::from_secs(5), rx).await {
            Ok(Ok(v)) => {
                info!("[tg][debug] get_file_path: получен ответ для {}: {}", file_id, v);
                // Ответ getFile — это объект file НАПРЯМУЮ (v["local"]),
                // НЕ обёрнутый в v["file"]["local"] (как в updateFile).
                let local = &v["local"];
                let completed = local["is_downloading_completed"].as_bool().unwrap_or(false);
                let path = local["path"].as_str().unwrap_or("").to_string();
                if completed && !path.is_empty() {
                    Ok(Some(path))
                } else {
                    Ok(None)
                }
            }
            Ok(Err(_)) => {
                info!("[tg][debug] get_file_path: канал закрыт для {}", file_id);
                self.pending_queries.lock().unwrap().remove(&extra);
                Ok(None)
            }
            Err(_) => {
                info!("[tg][debug] get_file_path: таймаут 5с для {}", file_id);
                self.pending_queries.lock().unwrap().remove(&extra);
                Ok(None)
            }
        }
    }

    /// Отправляет текстовое сообщение в чат (через sendMessage + inputMessageText).
    pub fn send_text(&self, chat_id: i64, text: &str) {
        self.send(&serde_json::json!({
            "@type": "sendMessage",
            "chat_id": chat_id,
            "input_message_content": {
                "@type": "inputMessageText",
                "text": { "@type": "formattedText", "text": text },
                "disable_web_page_preview": true,
                "clear_draft": true
            }
        }));
    }

    /// Отправляет локальный файл-картинку в чат (sendMessage + inputMessagePhoto).
    /// `path` — абсолютный путь к PNG/JPG на диске (например, сгенерированный SD).
    /// Использует `inputFileLocal` — TDLib сам подхватит и загрузит файл.
    pub fn send_photo(&self, chat_id: i64, path: &str) {
        self.send(&serde_json::json!({
            "@type": "sendMessage",
            "chat_id": chat_id,
            "input_message_content": {
                "@type": "inputMessagePhoto",
                "photo": {
                    "@type": "inputFileLocal",
                    "path": path
                },
                "width": 0,
                "height": 0,
                "caption": { "@type": "formattedText", "text": "" }
            }
        }));
    }

    /// Отправляет ГОЛОСОВОЕ сообщение (sendMessage + inputMessageVoiceNote).
    /// `path` — абсолютный путь к MP3/OGA на диске (синтез TTS).
    /// `duration_secs` — длительность в секундах (0 = TDLib сам разберётся,
    /// но для правильной визуализации волны лучше задать реальную).
    pub fn send_voice_note(&self, chat_id: i64, path: &str, duration_secs: i64) {
        self.send(&serde_json::json!({
            "@type": "sendMessage",
            "chat_id": chat_id,
            "input_message_content": {
                "@type": "inputMessageVoiceNote",
                "voice_note": {
                    "@type": "inputFileLocal",
                    "path": path
                },
                "duration": duration_secs,
                "waveform": "",
                "caption": { "@type": "formattedText", "text": "" }
            }
        }));
    }

    /// Запрашивает список активных чатов (getChats).
    /// Лимит — сколько чатов подгрузить сразу.
    pub fn get_chats(&self, limit: i64) {
        self.send(&serde_json::json!({
            "@type": "getChats",
            "offset_order": 9223372036854775807i64,
            "offset_chat_id": 0,
            "limit": limit
        }));
    }

    /// Отправляет статус "печатает..." в чат (sendChatAction + chatActionTyping).
    pub fn send_action_typing(&self, chat_id: i64) {
        self.send(&serde_json::json!({
            "@type": "sendChatAction",
            "chat_id": chat_id,
            "action": { "@type": "chatActionTyping" }
        }));
    }

    /// Открывает чат (openChat) — нужно перед загрузкой истории, иначе TDLib
    /// не держит сообщения локально и getChatHistory/searchChatMessages возвращают 0.
    pub fn open_chat(&self, chat_id: i64) {
        self.send(&serde_json::json!({
            "@type": "openChat",
            "chat_id": chat_id
        }));
    }

    /// Настраивает прокси для TDLib (нужен там, где Telegram режут по IP).
    ///
    /// Схема TDLib 1.8.62 (взяснена пробником probe_proxy):
    ///   1. `getProxies` → ищем существующий прокси по server:port;
    ///   2. `addProxy` с ВЛОЖЕННЫМ объектом "proxy" → ответ `addedProxy`
    ///      с целочисленным id (класса "proxySettings" в 1.8.62 нет!);
    ///   3. `enableProxy` с параметром `proxy_id` → прокси включается.
    ///
    /// Формат URL: `socks5://host:port` или `http://host:port`.
    pub async fn configure_proxy(&self, url: &str) -> anyhow::Result<()> {
        let url = url.trim();
        let (kind, hostport) = if let Some(rest) = url.strip_prefix("socks5://") {
            ("proxyTypeSocks5", rest.trim_end_matches('/').to_string())
        } else if let Some(rest) = url.strip_prefix("http://") {
            ("proxyTypeHttp", rest.trim_end_matches('/').to_string())
        } else {
            anyhow::bail!("не понимаю прокси '{url}' (нужен socks5:// или http://)");
        };
        let (host, port) = hostport
            .rsplit_once(':')
            .ok_or_else(|| anyhow::anyhow!("в прокси нет порта: {hostport}"))?;
        let port: i64 = port
            .parse()
            .map_err(|_| anyhow::anyhow!("порт прокси не число: {port}"))?;

        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        // 1) Ищем уже сохранённый прокси (они живут в базе TDLib).
        let extra = format!("getproxies_{nanos}");
        let list = self
            .query_td(
                serde_json::json!({ "@type": "getProxies" }),
                &extra,
                Duration::from_secs(20),
            )
            .await?;
        let proxies = list
            .get("proxies")
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default();
        let existing = proxies.iter().find_map(|p| {
            let id = p.get("id").and_then(|i| i.as_i64())?;
            let pr = p.get("proxy")?;
            let same = pr.get("server").and_then(|s| s.as_str()) == Some(host)
                && pr.get("port").and_then(|x| x.as_i64()) == Some(port);
            same.then_some(id)
        });

        // 2) Нет — добавляем.
        let proxy_id = match existing {
            Some(id) => {
                info!("[tg] прокси {host}:{port} уже есть в TDLib (id={id})");
                id
            }
            None => {
                let extra = format!("addproxy_{nanos}");
                let resp = self
                    .query_td(
                        serde_json::json!({
                            "@type": "addProxy",
                            "proxy": {
                                "server": host,
                                "port": port,
                                "type": { "@type": kind }
                            }
                        }),
                        &extra,
                        Duration::from_secs(20),
                    )
                    .await?;
                let id = resp
                    .get("id")
                    .and_then(|x| x.as_i64())
                    .ok_or_else(|| anyhow::anyhow!("addProxy вернул не прокси: {resp}"))?;
                info!("[tg] прокси добавлен в TDLib: {host}:{port} ({kind}), id={id}");
                id
            }
        };

        // 3) Включаем (иначе TDLib продолжает ходить напрямую).
        let extra = format!("enableproxy_{nanos}");
        let resp = self
            .query_td(
                serde_json::json!({ "@type": "enableProxy", "proxy_id": proxy_id }),
                &extra,
                Duration::from_secs(20),
            )
            .await?;
        if resp.get("@type").and_then(|v| v.as_str()) == Some("error") {
            anyhow::bail!("enableProxy: {resp}");
        }
        info!("[tg] прокси для Telegram включён: {host}:{port} ({kind})");
        Ok(())
    }

    /// Управляет статусом "в сети" (TDLib option "online"). Ошибкам TDLib
    /// тоже всё равно — option может отсутствовать, тогда просто не сработает.
    /// Используем: offline по умолчанию (Саша не хочет вечного "В сети"),
    /// online — только во время живого общения.
    pub fn set_online(&self, online: bool) {
        self.send(&serde_json::json!({
            "@type": "setOption",
            "name": "online",
            "value": { "@type": "optionValueBoolean", "value": online }
        }));
    }

    /// Планирует правку «опечатки»: после успешной отправки сообщения worker
    /// отредактирует его на `correct_text` через 1.4–2.8с. FIFO на чат.
    pub fn queue_typo_fix(&self, chat_id: i64, correct_text: String) {
        self.pending_edits.lock().unwrap().push_back((chat_id, correct_text));
    }

    /// getOption с ожиданием ответа (диагностика статуса и прочих опций).
    pub async fn get_option_async(&self, name: &str) -> anyhow::Result<Value> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let extra = format!("getopt_{name}_{nanos}");
        self.query_td(
            serde_json::json!({ "@type": "getOption", "name": name }),
            &extra,
            Duration::from_secs(8),
        )
        .await
    }

    /// Запрашивает историю сообщений чата (getChatHistory).
    /// `extra` — уникальный ID для корреляции ответа (TDLib вернёт его в `@extra`).
    pub fn get_chat_history(&self, chat_id: i64, limit: i64, from_message_id: i64, extra: &str) {
        self.send(&serde_json::json!({
            "@type": "getChatHistory",
            "chat_id": chat_id,
            "from_message_id": from_message_id,
            "offset": 0,
            "limit": limit,
            "only_local": false,
            "@extra": extra
        }));
    }

    /// Ищет/грузит сообщения чата (searchChatMessages). Агрессивнее getChatHistory —
    /// форсирует подгрузку старой истории с сервера. `offset` отрицательный = старые сообщения.
    pub fn search_chat_messages(
        &self,
        chat_id: i64,
        from_message_id: i64,
        offset: i64,
        limit: i64,
        extra: &str,
    ) {
        self.send(&serde_json::json!({
            "@type": "searchChatMessages",
            "chat_id": chat_id,
            "query": "",
            "sender_id": null,
            "from_message_id": from_message_id,
            "offset": offset,
            "limit": limit,
            "filter": null,
            "@extra": extra
        }));
    }

    /// Ставит реакцию (эмодзи) на конкретное сообщение.
    ///
    /// TDLib: `addMessageReaction chat_id:int53 message_id:int53
    /// reaction_type:ReactionType is_big:Bool update_recent_reactions:Bool`,
    /// где ReactionType = `reactionTypeEmoji emoji:string`.
    /// (Класса "addReaction" не существует — TDLib отвечает 400
    /// "Unknown class \"addReaction\"".)
    pub fn add_reaction(&self, chat_id: i64, message_id: i64, emoji: &str) {
        self.send(&serde_json::json!({
            "@type": "addMessageReaction",
            "chat_id": chat_id,
            "message_id": message_id,
            "reaction_type": { "@type": "reactionTypeEmoji", "emoji": emoji },
            "is_big": false,
            "update_recent_reactions": true
        }));
    }

    /// Общий запрос-ответ: шлёт метод с @extra и ждёт ответ через pending_queries.
    pub async fn query_td(&self, mut req: Value, extra: &str, timeout: Duration) -> anyhow::Result<Value> {
        let (tx, rx) = oneshot::channel::<Value>();
        self.pending_queries.lock().unwrap().insert(extra.to_string(), tx);
        req["@extra"] = serde_json::json!(extra);
        self.send(&req);
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(v)) => Ok(v),
            _ => anyhow::bail!("TDLib: таймаут/закрытый канал для {} ({})", req["@type"], extra),
        }
    }

    /// user_id инлайн-бота @gif (официальный GIF-поиск Telegram, без всяких
    /// сторонних API-ключей). Резолвится через searchPublicChat("gif")
    /// один раз и кэшится.
    async fn inline_gif_bot_id(&self) -> anyhow::Result<i64> {
        static BOT_PORTER_ID: tokio::sync::OnceCell<i64> = tokio::sync::OnceCell::const_new();
        let id = *BOT_PORTER_ID
            .get_or_try_init(|| async {
                let chat = self
                    .query_td(
                        serde_json::json!({ "@type": "searchPublicChat", "username": "gif" }),
                        "gifbotcfg",
                        Duration::from_secs(10),
                    )
                    .await?;
                chat["id"]
                    .as_i64()
                    .ok_or_else(|| anyhow::anyhow!("searchPublicChat(gif): нет id в ответе"))
            })
            .await?;
        Ok(id)
    }

    /// Ищет GIF через инлайн-бота @gif и ОТПРАВЛЯЕТ её в чат.
    /// Возвращает Ok(true) если гифка ушла, Ok(false) если ничего не нашлось.
    pub async fn search_and_send_gif(&self, chat_id: i64, query: &str) -> anyhow::Result<bool> {
        let bot_id = self.inline_gif_bot_id().await?;
        let results = self
            .query_td(
                serde_json::json!({
                    "@type": "getInlineQueryResults",
                    "chat_id": chat_id,
                    "bot_user_id": bot_id,
                    "query": query,
                    "offset": ""
                }),
                "gifsearch",
                Duration::from_secs(12),
            )
            .await?;
        if results["@type"] != "inlineQueryResults" {
            anyhow::bail!("ожидали inlineQueryResults, получили {}", results["@type"]);
        }
        let query_id = results["inline_query_id"].as_i64().unwrap_or(0);
        let result_id = results["results"]
            .as_array()
            .and_then(|arr| {
                // Предпочитаем анимации; fallback — любая картинка-гиф.
                let anim = arr.iter().find(|r| r["@type"] == "inlineQueryResultAnimation");
                anim.or_else(|| arr.first())
            })
            .and_then(|r| r["id"].as_str())
            .map(|s| s.to_string());
        let Some(rid) = result_id else {
            return Ok(false);
        };
        self.send(&serde_json::json!({
            "@type": "sendInlineQueryResultMessage",
            "chat_id": chat_id,
            "query_id": query_id,
            "result_id": rid,
            "hide_via_bot": false
        }));
        Ok(true)
    }

    /// Ставит аватарку аккаунта (setProfilePhoto). `path` — локальный файл
    /// PNG/JPEG. Используется при смене стиля аватара (одноразовая команда).
    pub fn set_profile_photo(&self, path: &str) {
        self.send(&serde_json::json!({
            "@type": "setProfilePhoto",
            "photo": {
                "@type": "inputChatPhotoStatic",
                "photo": { "@type": "inputFileLocal", "path": path }
            }
        }));
    }

    /// Загружает libtdjson и запускает worker-поток.
    ///
    /// `api_id`/`api_hash`/`database_dir` используются для синхронной
    /// инициализации (setTdlibParameters + setDatabaseEncryptionKey) через
    /// `td_json_client_execute` — это гарантирует применение ДО первого receive
    /// (worker-поток `send` для TDLib 1.8.62 иногда теряет поля, execute — нет).
    ///
    /// Возвращает `(client, UpdateLoop)` где `UpdateLoop.updates` — канал
    /// полученных update-ов (для авторизации и обработки сообщений).
    pub fn start(
        lib_path: &str,
        _api_id: i64,
        _api_hash: &str,
        _database_dir: &str,
    ) -> anyhow::Result<(TdClient, UpdateLoop)> {
        let lib = unsafe { Library::new(lib_path) }
            .map_err(|e| anyhow::anyhow!("не удалось загрузить {}: {}", lib_path, e))?;

        let create: Symbol<TdCreate> = unsafe { lib.get(b"td_json_client_create\0") }?;
        let send_fn: Symbol<TdSend> = unsafe { lib.get(b"td_json_client_send\0") }?;
        let recv_fn: Symbol<TdReceive> = unsafe { lib.get(b"td_json_client_receive\0") }?;
        let _exec_fn_sym: Symbol<TdExecute> = unsafe { lib.get(b"td_json_client_execute\0") }?;
        let destroy: Symbol<TdDestroy> = unsafe { lib.get(b"td_json_client_destroy\0") }?;

        // Создаём клиента в ЭТОМ (основном) потоке — handle получаем сразу.
        let handle = unsafe { create() };
        if handle.is_null() {
            anyhow::bail!("td_json_client_create вернул null");
        }
        // handle — сырой указатель (не Send). Кастуем в usize для передачи в поток.
        let handle_addr = handle as usize;

        // Параметры НЕ шлём здесь — TDLib 1.8.62 требует, чтобы send и receive
        // шли из ОДНОГО потока (worker). Инициализация происходит в authenticate()
        // через client.send() (попадает в worker-поток).
        info!("[tg] клиент создан, worker запущен");

        // Каналы: команды → worker, update-ы → вызывающий.
        let (cmd_tx, cmd_rx) = channel::<Option<String>>();
        let (upd_tx, upd_rx) = channel::<Value>();
        // Pending-загрузки файлов — разделяемый между вызывающим (download_file)
        // и worker-потоком (который ловит updateFile и заполняет oneshot).
        let downloads: Arc<Mutex<HashMap<i64, oneshot::Sender<String>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let downloads_for_worker = downloads.clone();
        // Pending-запросы (getFile и др.) — разделяемый между вызывающим и worker.
        let pending_queries: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let pending_queries_for_worker = pending_queries.clone();
        // Очередь правок опечаток (worker редактирует после успешной отправки).
        let pending_edits: Arc<Mutex<VecDeque<(i64, String)>>> =
            Arc::new(Mutex::new(VecDeque::new()));
        let pending_edits_for_worker = pending_edits.clone();

        // Worker-поток владеет handle + send/recv функциями.
        // Symbol из libloading не Send — извлекаем сырые адреса функций.
        // .so держим загруженным навсегда (leak), чтобы адреса функций были
        // валидны пока жив worker-поток. Для долгоживущего клиента это ок.
        let create_ptr = *create as usize;
        let send_ptr = *send_fn as usize;
        let recv_ptr = *recv_fn as usize;
        let destroy_ptr = *destroy as usize;
        let _lib_leak = Box::leak(Box::new(lib));
        let _ = _lib_leak;
        let _ = create_ptr; // create уже вызван

        // Копия cmd_tx для worker-потока (наши правки опечаток пойдут через него).
        let cmd_tx_worker = cmd_tx.clone();
        thread::spawn(move || {
            // Восстанавливаем типизированные указатели функций из сырых адресов.
            let send_fn: TdSend = unsafe { std::mem::transmute(send_ptr) };
            let recv_fn: TdReceive = unsafe { std::mem::transmute(recv_ptr) };
            let destroy_fn: TdDestroy = unsafe { std::mem::transmute(destroy_ptr) };
            let handle = handle_addr as *mut std::ffi::c_void;

            loop {
                // Ждём команду (блокирующе) или таймаут receive.
                match cmd_rx.recv_timeout(Duration::from_secs(1)) {
                    Ok(Some(cmd)) => {
                        let c_cmd = CString::new(cmd).unwrap_or_default();
                        unsafe { send_fn(handle, c_cmd.as_ptr()) };
                    }
                    Ok(None) => {
                        // Сигнал завершения: выходим из цикла, разрушаем клиента.
                        break;
                    }
                    Err(_) => {
                        // Таймаут — просто делаем receive (чтобы не блокировать вечно).
                    }
                }
                // Получаем накопившиеся update-ы (неблокирующе, таймаут 0).
                loop {
                    let raw = unsafe { recv_fn(handle, 0.0) };
                    if raw.is_null() {
                        break;
                    }
                    // ВАЖНО: TDLib сам владеет памятью, возвращаемой receive().
                    // НЕ вызываем free() — иначе double free.
                    let s = unsafe { CStr::from_ptr(raw).to_string_lossy().into_owned() };
                    if let Ok(v) = serde_json::from_str::<Value>(&s) {
                        // Обрабатываем updateFile: если кто-то ждёт этот file_id —
                        // отдаём локальный путь через oneshot и удаляем ожидание.
                        if v["@type"] == "updateFile" {
                            let fid = v["file"]["id"].as_i64().unwrap_or(0);
                            let local = &v["file"]["local"];
                            let completed = local["is_downloading_completed"].as_bool().unwrap_or(false);
                            let path = local["path"].as_str().unwrap_or("").to_string();
                            // Отдаём путь ТОЛЬКО когда файл реально докачан
                            // (is_downloading_completed). Иначе TDLib даёт
                            // промежуточный temp/-путь, который ещё не готов
                            // к чтению (image::open падает с "не удалось открыть").
                            #[allow(clippy::collapsible_if)]
                            if fid != 0 && completed && !path.is_empty() {
                                if let Some(tx) = downloads_for_worker.lock().unwrap().remove(&fid) {
                                    let _ = tx.send(path);
                                }
                            }
                        }
                        // Ответы на синхронные запросы (getFile и др.):
                        // если пришёл @extra, на который кто-то ждёт — отдаём весь ответ.
                        if let Some(extra) = v["@extra"].as_str() {
                            info!("[tg][debug] receive: пойман @extra={} type={}", extra, v["@type"]);
                            if let Some(tx) = pending_queries_for_worker
                                .lock()
                                .unwrap()
                                .remove(extra)
                            {
                                let _ = tx.send(v.clone());
                            }
                        }
                        // Правка «опечаток» (живость): сообщение успешно ушло —
                        // берём запланированный верный текст и редактирует через 1.4–2.8с.
                        if v["@type"] == "updateMessageSendSucceeded" {
                            let msg = &v["message"];
                            let chat_id = msg["chat_id"].as_i64().unwrap_or(0);
                            let message_id = msg["id"].as_i64().unwrap_or(0);
                            let planned = pending_edits_for_worker.lock().unwrap().pop_front();
                            if message_id != 0 {
                                if let Some((_, correct)) = planned {
                                    let cmd_tx_in_worker = cmd_tx_worker.clone();
                                    thread::spawn(move || {
                                        thread::sleep(Duration::from_millis(
                                            1400 + (rand::random::<u64>() % 1400),
                                        ));
                                        let edit = serde_json::json!({
                                            "@type": "editMessageText",
                                            "chat_id": chat_id,
                                            "message_id": message_id,
                                            "reply_markup": null,
                                            "input_message_content": {
                                                "@type": "inputMessageText",
                                                "text": { "@type": "formattedText", "text": correct },
                                                "disable_web_page_preview": true
                                            }
                                        });
                                        let _ = cmd_tx_in_worker.send(Some(edit.to_string()));
                                    });
                                }
                            }
                        }
                        if upd_tx.send(v).is_err() {
                            // Канал закрыт (вызывающий ушёл) — завершаемся.
                            break;
                        }
                    }
                }
            }
            // Destroy только здесь — один раз, в конце worker-потока.
            unsafe { destroy_fn(handle) };
        });

        Ok((TdClient { cmd_tx, downloads, pending_queries, pending_edits }, UpdateLoop { updates: upd_rx }))
    }

    /// Синхронное выполнение (для setTdlibParameters до запуска receive-цикла).
    /// Используется редко; в основном полагаемся на `send` + канал update-ов.
    pub fn execute(&self, _json: &Value) -> Option<Value> {
        // execute требует тот же handle, что и receive — поэтому делаем через
        // worker. Но для простоты пока не используем (setTdlibParameters шлём send-ом).
        None
    }
}

impl Drop for TdClient {
    fn drop(&mut self) {
        // Шлём сигнал завершения worker-потоку. Он сам вызовет destroy.
        let _ = self.cmd_tx.send(None);
    }
}

/// Цикл получения update-ов из worker-потока.
pub struct UpdateLoop {
    pub updates: Receiver<Value>,
}

/// Строит параметры TDLib (setTdlibParameters).
/// ПРИМЕЧАНИЕ: TDLib 1.8.62 ожидает ПЛОСКУЮ структуру (поля на верхнем уровне,
/// без вложенного `parameters`), иначе игнорирует их (api_id=0).
pub fn build_td_params(api_id: i64, api_hash: &str, database_dir: &str, lang: &str) -> Value {
    serde_json::json!({
        "@type": "setTdlibParameters",
        "use_test_dc": false,
        "api_id": api_id,
        "api_hash": api_hash,
        "database_directory": database_dir,
        "files_directory": "",
        "use_file_database": false,
        "use_chat_info_database": false,
        "use_message_database": true,
        "use_secret_chats": false,
        "system_language_code": lang,
        "device_model": "furinya",
        "system_version": "linux",
        "application_version": "0.1.0",
        "enable_storage_optimizer": true,
        "ignore_file_names": true
    })
}

// libloading Library должен жить, пока живы Symbol-ы в worker-потоке.
// Храним его в TdClient, но worker берёт только сырые указатели функций.
// Чтобы Library не drop-нулся раньше worker-потока, держим его в Arc.
// (упрощённо: Library drop-ается при drop TdClient, а worker уже завершён
//  к моменту drop благодаря сигналу None в Drop).
unsafe impl Send for TdClient {}
unsafe impl Sync for TdClient {}
