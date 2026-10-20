//! OpenAI-совместимый LLM клиент (миграция с OpenAIChatImpl C++).
//!
//! Поддерживает: chat completion (non-streaming + streaming), tool calls.
//! Работает поверх reqwest + tokio. Без unsafe.

use furinya_core::Config;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("HTTP ошибка запроса: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("ошибка десериализации ответа: {0}")]
    Json(#[from] serde_json::Error),
    #[error("LLM API вернул ошибку: {0}")]
    Api(String),
    #[error("поток завершился без контента")]
    StreamClosed,
}

/// Служебные префиксы строк в structured-выводе kuni (вырезаются из чистого ответа).
const KUNI_META_PREFIXES: &[&str] = &[
    "- thought:",
    "- emotion:",
    "- to remember:",
    "- intention:",
    "- tool:",
    "**Оценка",
    "**Plan",
    "**План",
    "**Решение",
    "**Выбор",
    "✅",
    "❌",
];

/// Native reasoning из поля `reasoning_content` (Gemma / OpenAI-o1 / DeepSeek
/// style). Это САМОСТОЯТЕЛЬНОЕ поле ответа — модель пишет его сама, мы лишь
/// перехватываем. НЕ пытаемся заставить модель оборачивать reasoning в теги:
/// почти любая современная модель с включённым reasoning выдаёт его именно так.
///
/// Возвращает `None`, если строка пустая/сплошь-whitespace.
pub fn extract_thinking(reasoning: &str) -> Option<String> {
    let cleaned = reasoning.trim().to_string();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Fallback-парсер: ищет legacy-служебные строки (`- thought:`, `- emotion:` и т.п.)
/// внутри обычного `content`, если модель по какой-то причине не выдала
/// `reasoning_content` (старые форматы / kuni-совместимость).
pub fn extract_thinking_from_content(raw: &str) -> Option<String> {
    let mut out = String::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if KUNI_META_PREFIXES.iter().any(|p| trimmed.starts_with(p)) {
            out.push_str(trimmed);
            out.push('\n');
        }
    }
    let cleaned = out.trim().to_string();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}
///
/// kuni форматирует ответ как набор служебных строк (`- thought:`, `- emotion:`,
/// `[Call tool: ...]` и т.п.). Эта функция отбрасывает их, возвращая только
/// человекочитаемый текст. Если после очистки ничего не осталось (модель ушла
/// в чистый tool-call или написала только раздумье), возвращает `None`.
///
/// Дополнительно отсекает строки-маркеры РАЗДУМЬЯ (reasoning), которые Gemma
/// иногда шлёт как plain content (не в `reasoning_content`): `💭`, "The user
/// asked/is asking", "My plan", "Action Plan", "Tool selection" и т.п. Иначе
/// сырое раздумье улетает в чат вместо ответа.
pub fn extract_clean_reply(raw: &str) -> Option<String> {
    let mut out = String::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if KUNI_META_PREFIXES.iter().any(|p| trimmed.starts_with(p)) {
            continue;
        }
        if trimmed.starts_with("[Call tool:") || trimmed.starts_with("[call tool:") {
            continue;
        }
        // Маркеры раздумья (reasoning), которые не должны попадать в ответ.
        if trimmed.starts_with('💭')
            || trimmed.starts_with("The user asked")
            || trimmed.starts_with("The user is asking")
            || trimmed.starts_with("My action plan")
            || trimmed.starts_with("Action Plan")
            || trimmed.starts_with("Tool selection")
            || trimmed.starts_with("Query formulation")
            || trimmed == "Plan:"
            || trimmed.starts_with("I will use")
        {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    let cleaned = out.trim().to_string();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

/// Распознанный вызов инструмента из сырого вывода модели (формат kuni).
#[derive(Debug, Clone)]
pub struct KuniToolCall {
    pub name: String,
    /// Сырой JSON-аргумент внутри скобок, напр. `{"query":"..."}`.
    pub args_json: String,
}

/// Ищет вызовы инструментов в сыром выводе модели.
///
/// Поддерживает два формата (kuni / Gemma через OAI API):
///   1. `[Call tool: name({json})]`
///   2. `<|tool_call>call:name{json}`
///
/// Возвращает пустой вектор, если вызовов нет. JSON внутри скобок может
/// содержать вложенные `{}`, поэтому ищем парную закрывающую скобку.
/// Поиск ведётся через `str::find`, что гарантирует корректные UTF-8 границы.
pub fn parse_tool_calls(raw: &str) -> Vec<KuniToolCall> {
    let mut calls = Vec::new();
    let mut search_from = 0;
    while search_from < raw.len() {
        // Формат 1: [Call tool: name({
        let fmt1 = raw[search_from..].find("[Call tool:");
        // Формат 2: <|tool_call>call: name{
        let fmt2 = raw[search_from..].find("<|tool_call>");

        // Берём ближайший маркер.
        let (abs_marker, prefix_len, mode) = match (fmt1, fmt2) {
            (Some(a), Some(b)) if a <= b => (search_from + a, "[Call tool:".len(), 1u8),
            (Some(a), _) => (search_from + a, "[Call tool:".len(), 1u8),
            (_, Some(b)) => (search_from + b, "<|tool_call>".len(), 2u8),
            (None, None) => break,
        };

        let after_prefix = abs_marker + prefix_len;
        let rest = &raw[after_prefix..];

        // Имя: до '(' (формат 1) или до '{' (Gemma пишет `name{json}` без скобок).
        let (name, open) = if mode == 1 {
            // Gemma иногда пишет `[Call tool: ask{...}]` (без '('), иногда
            // `[Call tool: ask({...})`. Берём первый из разделителей.
            let paren = rest.find('(');
            let brace = rest.find('{');
            let sep = match (paren, brace) {
                (Some(p), Some(b)) => Some(p.min(b)),
                (Some(p), None) => Some(p),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            };
            let Some(sep) = sep else { break };
            let is_brace = brace == Some(sep);
            let name = rest[..sep].trim().to_string();
            let open = if is_brace {
                after_prefix + sep // '{' сама — аргументы начинаются со следующего символа
            } else {
                after_prefix + sep + 1 // после '('
            };
            (name, open)
        } else {
            // формат 2: "call:name{json}" — выкусываем "call:"
            let after_call = rest
                .find("call:")
                .map(|i| after_prefix + i + "call:".len())
                .unwrap_or(after_prefix);
            let r2 = &raw[after_call..];
            let Some(brace) = r2.find('{') else { break };
            (r2[..brace].trim().to_string(), after_call + brace)
        };

        // Ищем парную '}' (учитывая вложенные {}).
        let bytes = raw.as_bytes();
        let mut depth = 0i32;
        let mut end = None;
        let mut j = open;
        while j < bytes.len() {
            match bytes[j] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(j);
                        break;
                    }
                }
                _ => {}
            }
            j += 1;
        }
        match end {
            Some(e) => {
                let mut args_json = raw[open + 1..e].to_string();
                // Gemma через OAI API иногда оборачивает строки в теги `<|"|>`,
                // `<｜...｜>` и т.п. — выкусываем их, чтобы JSON парсился.
                args_json = strip_angle_tags(&args_json);
                calls.push(KuniToolCall { name, args_json });
                search_from = e + 1;
            }
            None => break,
        }
    }
    calls
}

/// Вырезает теги вида `<|...|>`, `<｜...｜>` (Gemma chat-template артефакты),
/// оставляя чистое содержимое. Используется чтобы JSON-аргументы инструментов
/// парсились после того, как модель обернёт строки в такие маркеры.
fn strip_angle_tags(s: &str) -> String {
    // Простой проход: удаляем подстроки, начинающиеся на <| и заканчивающиеся |>.
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
            // ищем закрывающий |>
            if let Some(end) = s[i + 2..].find("|>") {
                i = i + 2 + end + 2; // пропускаем весь тег
                continue;
            }
        }
        // безопасно копируем один UTF-8 символ
        let ch_len = utf8_len(bytes[i]);
        if i + ch_len <= bytes.len() {
            out.push_str(&s[i..i + ch_len]);
        }
        i += ch_len;
    }
    out
}

#[inline]
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Один вызов инструмента от модели.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// Аргументы в виде JSON-строки (спецификация OpenAI).
    pub arguments: String,
}

/// Содержимое сообщения. Поддерживает как обычный текст, так и
/// multimodal-части (текст + изображение) для vision-моделей.
///
/// Сериализуется прозрачно: `Text(s)` → `"s"` (строка, как раньше),
/// `Parts(v)` → JSON-массив content-part'ов OpenAI-формата. Десериализация
/// untagged — модель может вернуть либо строку, либо массив (на всякий).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// Один кусок multimodal-сообщения (текст или картинка).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Текстовый кусок.
    #[serde(rename = "text")]
    Text { text: String },
    /// Картинка (base64 data-URI, как требует OpenAI/chat-completions).
    #[serde(rename = "image_url")]
    Image { image_url: ImageUrl },
}

/// Ссылка на картинку (data URI вида `data:image/jpeg;base64,....`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

/// Сообщение в истории чата.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
    /// Результат исполнения инструмента (role = tool).
    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
    /// Результат инструмента с привязкой к вызову (tool_call_id/name).
    /// OpenAI-совместимые API (в т.ч. Mistral) требуют связи tool-ответа
    /// с предшествующим assistant.tool_calls.
    pub fn tool_result(
        tool_call_id: Option<String>,
        name: Option<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent::Text(content.into()),
            tool_calls: None,
            tool_call_id,
            name,
        }
    }
    /// Assistant-сообщение с вызовами инструментов. Обязательный элемент
    /// порядка assistant(tool_calls) → tool(...) для Mistral: иначе API
    /// отвечает 400 "Unexpected role 'tool' ..." (код 3230).
    pub fn assistant_tool_calls(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            tool_calls: Some(calls),
            tool_call_id: None,
            name: None,
        }
    }
    /// User-сообщение из нескольких частей (текст + картинка) для vision.
    pub fn user_parts(parts: Vec<ContentPart>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Parts(parts),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
}

/// Описание функции для tool calling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    /// JSON Schema параметров.
    pub parameters: Value,
}

impl Tool {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            tool_type: "function".into(),
            function: ToolFunction {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

/// Набор инструментов для агента (Telegram-режим).
///
/// Ключевая идея: модель САМА дробит ответ на блоки и вызывает
/// `send_telegram_message` с массивом частей (`parts`) — рантайм просто
/// пересылает их с задержкой. Никакого парсинга `\n\n` на стороне Rust.
/// Это надёжнее, чем неявные теги разделения.
pub fn build_tools() -> Vec<Tool> {
    let str_arr = serde_json::json!({
        "type": "object",
        "properties": {
            "parts": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Сообщение, разбитое на несколько коротких частей. Каждая часть — отдельное сообщение в Telegram."
            }
        },
        "required": ["parts"]
    });
    let emoji_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": "Emoji для реакции (из разрешённого набора: 👍 👎 ❤️ 🔥 🥰 👏 😁 🤔 и т.д.)"
            }
        },
        "required": ["emoji"]
    });
    let query_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Вопрос к дневнику памяти (RAG-поиск релевантных воспоминаний)"
            }
        },
        "required": ["query"]
    });
    let text_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Текст для сохранения в дневник (консолидация памяти)"
            }
        },
        "required": ["text"]
    });

    let gif_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Поисковый запрос к GIF: 2-3 коротких слова АНГЛИЙСКИМ ('happy dance', 'shocked anime', 'wink wink', 'party cat')."
            }
        },
        "required": ["query"]
    });
    let gen_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Позитивный промпт для генерации изображения (детально: что именно нарисовать, стиль, настроение)."
            },
            "negative_prompt": {
                "type": "string",
                "description": "Что исключить из изображения (артефакты, размытость, лишние объекты). Опционально."
            },
            "nsfw": {
                "type": "boolean",
                "description": "Только если Саша ЯВНО попросил откровенное фото: true. Для обычных фото — false или пропуск поля."
            }
        },
        "required": ["prompt"]
    });
    let empty_schema = serde_json::json!({
        "type": "object",
        "properties": {}
    });

    vec![
        Tool::new(
            "stay_quiet",
            "ПРОМОЛЧАТЬ — ничего не отправлять в чат. Вызывай, когда САМА решила не писать: \
             в idle-моменте (Саша давно молчит и тебе нечего сказать), или когда нечего ответить. \
             Это НОРМАЛЬНЫЙ выбор (не ошибка), но и писать первой — тоже нормально, если тебе \
             хочется поделиться чем-то своим. Главное — не писать «просто чтобы напомнить о себе», \
             «ты тут?», «жду ответа»: это nagging, так не делают. Никаких сообщений при этом \
             в чат не уходит, ход считается завершённым.",
            empty_schema,
        ),
        Tool::new(
            "send_telegram_message",
            "Отправить ответ пользователю в Telegram. Передай массив частей (parts) — каждая часть станет отдельным сообщением. Дроби ответ сам, по смыслу. Хитрость: одиночный эмодзи, отправленный ЕДИНСТВЕННОЙ частью, Telegram показывает большой (jumbo) — так делают акцент на эмоцию.",
            str_arr,
        ),
        Tool::new(
            "react_with_emoji",
            "Поставить emoji-реакцию на сообщение собеседника. БЫСТРАЯ эмоциональная \
             реакция ВМЕСТО текста (когда нечего/незачем писать) ИЛИ ДОПОЛНЕНИЕ к ответу. \
             Выбирай ОБДУМАННО по настроению сообщения: смех — 😂, \
             любовь/умиление — ❤️🥰, грусть/сочувствие — 🥺😢, \
             удивление — 😮🤯, согласие — 👍, «огонь/круто» — 🔥, \
             что-то обдумываешь — 🤔. \
             НЕ ставь вовсе, если реакция не уместна (напр. серьёзные/грустные темы). \
             Отправляй её В ОДНОМ ходу с send_telegram_message, когда хочешь и отреагировать, и ответить.",
            emoji_schema,
        ),
        Tool::new(
            "ask",
            "Спросить дневник памяти о пользователе/событиях/фактах перед ответом.",
            query_schema,
        ),
        Tool::new(
            "diary_page",
            "Сохранить факты/воспоминания в дневник (долговременную память). ВНИМАНИЕ: в Telegram-чате НЕ вызывать — только в консольном режиме.",
            text_schema.clone(),
        ),
        Tool::new(
            "send_gif",
            "Отправить уместную смешную GIF-анимацию из встроенного поиска Telegram \
             (без всяких ссылок). Используй ИЗРЕДКА — когда обычного текста/эмодзи \
             явно мало: смешной момент, шок, злорадство/хехе, празднование, \
             обнимашки, прикольно-мемный момент. НЕ для грустных, серьёзных и \
             интимных тем. Можно вызвать в ОДНОМ ходу с send_telegram_message.",
            gif_schema,
        ),
        Tool::new(
            "generate_image",
            "Сделать фото/нарисовать изображение и отправить его в чат. ТЫ МОЖЕШЬ \
             «сфотографироваться»: твой аниме-образ уже вшит в генерацию, в prompt \
             описывай ТОЛЬКО сцену (позу, настроение, место, во что одета — по желанию). \
             Вызывай если Саша просит «сфоткайся», «покажи себя», прислать фото, \
             или рисует что-то ещё. Рендер занимает ~3 минуты. ВАЖНО про порядок: можно \
             сначала отправить короткую живую реплику («сейчас, минутку 📸») и СРАЗУ \
             после неё, следующим вызовом, вызвать generate_image — фото уйдёт в чат \
             отдельным сообщением. Если вызвала generate_image, отдельная реплика потом \
             не нужна.",
            gen_schema,
        ),
        Tool::new(
            "send_voice_message",
            "Отправить ГОЛОСОВОЕ сообщение с твоим голосом (синтез речи TTS). Вызывай если пользователь просит написать голосом/записать голосовое, или когда уместно озвучить короткий ответ. В аргументе text — ЧИСТЫЙ текст для озвучки: без эмодзи, списков, служебных строк и смайлов. Не длиннее 3-4 предложений.",
            text_schema.clone(),
        ),
    ]
}

/// Настройки сэмплинга, маппятся из Config.misc.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    /// Доп. параметры ("none" => не передаём).
    pub top_p: Option<f32>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

/// Тело запроса к /chat/completions.
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    /// Форсируем вызов инструмента: модель ОБЯЗАНА вернуть tool_call
    /// (обычно send_telegram_message), а не сырой текст. Это устраняет
    /// неявный парсинг сырого вывода и утечку раздумий в чат.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

/// Частичное сообщение из streaming-ответа.
#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
    #[serde(default)]
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    /// Native reasoning (Gemma/OpenAI-o1/DeepSeek). Поле может отсутствовать
    /// в некоторых провайдерах — тогда None и мы просто его игнорируем.
    #[serde(default)]
    reasoning_content: Option<String>,
    /// Native tool calls (OpenAI format). В стриме приходит кусками:
    /// каждый элемент имеет `index`, `id`, `function.name`, `function.arguments`
    /// (arguments нарастают). Аккумулируем в chat_stream.
    #[serde(default)]
    tool_calls: Option<Vec<StreamToolCallDelta>>,
}

/// Кусок tool_call из streaming delta (OpenAI format).
#[derive(Debug, Deserialize, Default)]
struct StreamToolCallDelta {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamFunctionDelta>,
}

#[derive(Debug, Deserialize, Default)]
struct StreamFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Полный ответ (non-streaming).
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: Message,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Клиент LLM. Безопасен для клонирования (reqwest::Client внутри Arc).
///
/// Поддерживает несколько моделей: первая — основная, следующие — fallback.
/// Если основная отдала HTTP-ошибку (429/5xx/таймаут при открытии соединения),
/// запрос повторяется на резервной. Это нужно для миграции на Mistral API:
/// mistral-large-2512 (основная) → ministral-14b-2512 (резерв).
#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    base_url: String,
    bearer_key: String,
    /// Модели в порядке приоритета: [основная, fallback...].
    models: Vec<String>,
    sampling: SamplingParams,
}

/// Отправка запроса с ретраем при 429 (бесплатный тариф Mistral жёстко лимитирует
/// RPM, причём лимит ПО-МОДЕЛЬНЫЙ: fallback на ministral этой квоты не касается).
/// Пауз мало: 2с → 4с, максимум 3 попытки, дальше отдаём ошибку наверх —
/// цикл по моделям переключит на fallback.
async fn send_with_rate_limit_retry(
    builder: reqwest::RequestBuilder,
) -> Result<reqwest::Response, reqwest::Error> {
    const MAX_ATTEMPTS: u32 = 3;
    for attempt in 1..=MAX_ATTEMPTS {
        // RequestBuilder не реюзается — клон на каждую попытку.
        let resp = builder
            .try_clone()
            .expect("POST RequestBuilder клонируется")
            .send()
            .await?;
        if resp.status().as_u16() == 429 && attempt < MAX_ATTEMPTS {
            let wait = std::time::Duration::from_secs(1u64 << attempt); // 2с, 4с
            eprintln!(
                "[llm] 429 rate limit — жду {wait:?} и ретраю (попытка {attempt}/{MAX_ATTEMPTS})"
            );
            tokio::time::sleep(wait).await;
            continue;
        }
        return Ok(resp);
    }
    unreachable!()
}

/// Ключ из окружения ТОЛЬКО для Mistral API (чтобы чужой конфиг на другой
/// провайдер не унаследовал mistral-ключ). Пустая строка = нет ключа.
fn env_mistral_key(base_url: &str) -> String {
    if !base_url.contains("api.mistral.ai") {
        return String::new();
    }
    std::env::var("MISTRAL_API_KEY")
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty())
        .unwrap_or_default()
}

fn config_base_is_mistral(cfg: &Config) -> bool {
    cfg.general.llm.endpoint.base_url.contains("api.mistral.ai")
}


/// Приводит историю сообщений к виду, который принимает Mistral (и другие
/// OpenAI-совместимые бэкенды).
///
/// Правила (каждое — реальная ошибка 400 от Mistral):
/// 1. Последнее сообщение не должно быть «голым» assistant без tool_calls:
///    `Expected last role User or Tool (or Assistant with prefix True)`.
/// 2. Сообщение с ролью `tool` обязано идти сразу за assistant'ом с
///    tool_calls — иначе «осиротевший» tool ломает порядок (400 3230).
/// 3. У assistant'а с tool_calls должны быть ответы-tool; если их нет —
///    вызовы снимаются (иначе API ждёт tool-результат, которого не будет).
/// 4. Первой не может быть `tool` (такие висят в начале — ломают порядок).
/// 5. Если не осталось ни одного не-системного сообщения — добавляем
///    пустой user, иначе запрос бессмысленен.
pub fn sanitize_messages(messages: Vec<Message>) -> Vec<Message> {
// 4) ведущие tool убираем
let mut out: Vec<Message> = messages
    .into_iter()
    .skip_while(|m| m.role == Role::Tool)
    .collect();

// 2) и 3) выкидываем осиротевшие tool и чиним assistant без ответов
let mut cleaned: Vec<Message> = Vec::with_capacity(out.len());
let mut i = 0usize;
while i < out.len() {
    let m = out[i].clone();
    if m.role == Role::Tool {
        // tool без предшествующего assistant(tool_calls) — выкидываем.
        let has_owner = cleaned
            .last()
            .map(|p| p.role == Role::Assistant && p.tool_calls.is_some())
            .unwrap_or(false);
        if !has_owner {
            i += 1;
            continue;
        }
        cleaned.push(m);
        i += 1;
        continue;
    }
    if m.role == Role::Assistant {
        if let Some(calls) = m.tool_calls.clone() {
            if !calls.is_empty() {
                // Считаем, сколько tool-ответов идёт сразу после.
                let mut j = i + 1;
                let mut answers = 0usize;
                while j < out.len() && out[j].role == Role::Tool {
                    answers += 1;
                    j += 1;
                }
                if answers == 0 {
                    // 3) ответов нет — снимаем вызовы, оставляем текст.
                    let mut plain = m.clone();
                    plain.tool_calls = None;
                    cleaned.push(plain);
                    i += 1;
                    continue;
                }
            }
        }
    }
    cleaned.push(m);
    i += 1;
}
out = cleaned;

// 1) последнее сообщение — не assistant (Mistral требует user/tool в конце;
//    assistant допустим только с tool_calls, но тогда нужны ответы-tool).
//    Снимаем ВСЕ хвостовые assistant-сообщения: их может быть несколько
//    (её реплика в истории + неотвеченный вызов инструмента).
while out.last().map(|m| m.role == Role::Assistant).unwrap_or(false) {
    out.pop();
}
// Замыкающие system тоже убираем (последним должен быть user/tool).
while out.last().map(|m| m.role == Role::System).unwrap_or(false) {
    out.pop();
}
// 5) пустой запрос — подставляем минимальный user
if out.iter().all(|m| m.role == Role::System) {
    out.push(Message::user("(продолжи разговор)"));
}
out
}


impl LlmClient {
    /// Прямой конструктор (для тестов / нестандартных сценариев).
    ///
    /// Fallback ключа из окружения: если передан пустой bearer_key, а база —
    /// Mistral API, подхватывается `MISTRAL_API_KEY` из env.
    pub fn new(
        base_url: impl Into<String>,
        bearer_key: impl Into<String>,
        model: impl Into<String>,
        sampling: SamplingParams,
    ) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(900))
            .build()
            .expect("reqwest client build");
        let base = base_url.into().trim_end_matches('/').to_string();
        let mut key = bearer_key.into();
        if key.is_empty() {
            key = env_mistral_key(&base);
        }
        Self {
            http,
            base_url: base,
            bearer_key: key,
            models: vec![model.into()],
            sampling,
        }
    }

    /// Строит клиент из конфига ядра.
    pub fn from_config(cfg: &Config) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(900))
            .build()?;
        let sampling = SamplingParams {
            temperature: cfg.misc.llm_temperature,
            top_p: parse_opt_f32(&cfg.misc.llm_top_p),
            top_k: None,
            min_p: parse_opt_f32(&cfg.misc.llm_min_p),
            presence_penalty: parse_opt_f32(&cfg.misc.presence_penalty),
            repetition_penalty: parse_opt_f32(&cfg.misc.repetition_penalty),
        };
        let mut models = vec![cfg.general.llm.endpoint.model.clone()];
        let fallback = cfg.general.llm.endpoint.fallback_model.trim().to_string();
        if !fallback.is_empty() && fallback != models[0] {
            models.push(fallback);
        }
        let mut key = cfg.general.llm.endpoint.bearer_key.clone();
        if key.is_empty() && config_base_is_mistral(cfg) {
            key = env_mistral_key("https://api.mistral.ai");
        }
        Ok(Self {
            http,
            base_url: cfg
                .general
                .llm
                .endpoint
                .base_url
                .trim_end_matches('/')
                .to_string(),
            bearer_key: key,
            models,
            sampling,
        })
    }

    /// Значение `tool_choice` для текущего провайдера.
    /// OpenAI/llama.cpp: "required". Mistral API: "any" (их вариант forced tool use).
    fn tool_choice_value(&self) -> String {
        if self.base_url.contains("api.mistral.ai") {
            "any".to_string()
        } else {
            "required".to_string()
        }
    }
/// Non-streaming запрос. Возвращает первый choice.
    ///
    /// При ошибке основной модели повторяет запрос на fallback-моделях.
    pub async fn chat(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        for (i, model) in self.models.iter().enumerate() {
            if i > 0 {
                eprintln!(
                    "[llm] fallback: переключаюсь на модель '{model}' после сбоя предыдущей"
                );
            }
            match self.chat_single_model(model, messages.clone(), tools.clone()).await {
                Ok(resp) => return Ok(resp),
                Err(e) if i + 1 < self.models.len() => {
                    eprintln!("[llm] основная модель '{model}' не ответила: {e}");
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!("модели не пусты")
    }

    async fn chat_single_model(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<Tool>>,
    ) -> Result<ChatResponse, LlmError> {
        // tool_choice шлём ТОЛЬКО вместе с tools: Mistral отвечает 400
        // ("No tools provided for tool choice `any`") на tool_choice без tools.
        let tool_choice = tools.as_ref().map(|_| self.tool_choice_value());
        // Порядок сообщений приводим к допустимому для Mistral (400 3230).
        let messages = sanitize_messages(messages);
        let req = ChatRequest {
            model: model.to_string(),
            messages,
            tools,
            tool_choice,
            temperature: Some(self.sampling.temperature),
            top_p: self.sampling.top_p,
            stream: Some(false),
        };
        let mut builder = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .json(&req);
        if !self.bearer_key.is_empty() {
            builder = builder.bearer_auth(&self.bearer_key);
        }
        let resp = send_with_rate_limit_retry(builder).await?;
        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api(body));
        }
        let parsed: ChatResponse = resp.json().await?;
        Ok(parsed)
    }

    /// Streaming запрос. Вызывает `on_delta` для каждого куска контента и
    /// `on_reasoning` для каждого куска native reasoning (если модель его шлёт).
    ///
    /// При ошибке открытия потока основной моделью повторяет на fallback-моделях.
    /// Если поток ОКРЫЛСЯ и оборвался посреди — fallback НЕ выполняется
    /// (контент уже ушёл вызвавшему).
    pub async fn chat_stream<F, G>(
        &self,
        messages: Vec<Message>,
        tools: Option<Vec<Tool>>,
        mut on_delta: F,
        mut on_reasoning: G,
    ) -> Result<Vec<ToolCall>, LlmError>
    where
        F: FnMut(&str),
        G: FnMut(&str),
    {
        for (i, model) in self.models.iter().enumerate() {
            if i > 0 {
                eprintln!(
                    "[llm] fallback: переключаюсь на модель '{model}' после сбоя предыдущей"
                );
            }
            match self
                .chat_stream_single_model(model, messages.clone(), tools.clone(), &mut on_delta, &mut on_reasoning)
                .await
            {
                Ok(acc) => return Ok(acc),
                Err(e) if i + 1 < self.models.len() => {
                    eprintln!("[llm] основная модель '{model}' не ответила (стрим): {e}");
                }
                Err(e) => return Err(e),
            }
        }
        unreachable!("модели не пусты")
    }

    async fn chat_stream_single_model<F, G>(
        &self,
        model: &str,
        messages: Vec<Message>,
        tools: Option<Vec<Tool>>,
        on_delta: &mut F,
        on_reasoning: &mut G,
    ) -> Result<Vec<ToolCall>, LlmError>
    where
        F: FnMut(&str),
        G: FnMut(&str),
    {
        // tool_choice только при наличии tools (Mistral: 400 иначе).
        let tool_choice = tools.as_ref().map(|_| self.tool_choice_value());
        // Порядок сообщений приводим к допустимому для Mistral (400 3230).
        let messages = sanitize_messages(messages);
        let req = ChatRequest {
            model: model.to_string(),
            messages,
            tools,
            tool_choice,
            temperature: Some(self.sampling.temperature),
            top_p: self.sampling.top_p,
            stream: Some(true),
        };
        let mut builder = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .json(&req);
        if !self.bearer_key.is_empty() {
            builder = builder.bearer_auth(&self.bearer_key);
        }
        let resp = send_with_rate_limit_retry(builder).await?;
        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(LlmError::Api(body));
        }
        use futures_util::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        // Аккумулятор native tool_calls (по index из стрима).
        let mut acc: Vec<ToolCall> = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let text = String::from_utf8_lossy(&chunk);
            buf.push_str(&text);
            // SSE: строки "data: {...}\n\n"
            while let Some(nl) = buf.find('\n') {
                let line: String = buf.drain(..=nl).collect();
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Some(payload) = line.strip_prefix("data:") {
                    let payload = payload.trim();
                    if payload == "[DONE]" {
                        return Ok(acc);
                    }
                    if let Ok(chunk) = serde_json::from_str::<StreamChunk>(payload) {
                        for choice in &chunk.choices {
                            if let Some(ref content) = choice.delta.content {
                                on_delta(content);
                            }
                            if let Some(ref reasoning) = choice.delta.reasoning_content {
                                on_reasoning(reasoning);
                            }
                            // Аккумулируем native tool_calls по index.
                            if let Some(ref tcs) = choice.delta.tool_calls {
                                for tc in tcs {
                                    let idx = tc.index;
                                    if acc.len() <= idx {
                                        // Расширяем до нужного индекса (пустыми заглушками).
                                        while acc.len() <= idx {
                                            acc.push(ToolCall {
                                                id: String::new(),
                                                call_type: "function".into(),
                                                function: FunctionCall {
                                                    name: String::new(),
                                                    arguments: String::new(),
                                                },
                                            });
                                        }
                                    }
                                    let slot = &mut acc[idx];
                                    if let Some(id) = &tc.id {
                                        slot.id = id.clone();
                                    }
                                    if let Some(fn_delta) = &tc.function {
                                        if let Some(name) = &fn_delta.name {
                                            slot.function.name.push_str(name);
                                        }
                                        if let Some(args) = &fn_delta.arguments {
                                            slot.function.arguments.push_str(args);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(acc)
    }
}

/// Парсит "none" => None, иначе f32.
fn parse_opt_f32(s: &str) -> Option<f32> {
    let t = s.trim();
    if t.eq_ignore_ascii_case("none") || t.is_empty() {
        None
    } else {
        t.parse::<f32>().ok()
    }
}

#[cfg(test)]
mod sanitize_tests {
    use super::*;

    fn assistant_text(s: &str) -> Message {
        Message::assistant(s)
    }
    fn assistant_with_call(s: &str) -> Message {
        let mut m = Message::assistant(s);
        m.tool_calls = Some(vec![ToolCall {
            id: "c1".into(),
            call_type: "function".into(),
            function: FunctionCall {
                name: "send_telegram_message".into(),
                arguments: "{}".into(),
            },
        }]);
        m
    }
    fn tool_result(s: &str) -> Message {
        Message::tool_result(
            Some("c1".to_string()),
            Some("send_telegram_message".to_string()),
            s.to_string(),
        )
    }

    #[test]
    fn drops_trailing_bare_assistant() {
        // Реальный кейс: хвост истории кончается её собственной репликой →
        // Mistral: "Expected last role User or Tool ... but got assistant".
        let msgs = vec![
            Message::system("s"),
            Message::user("привет"),
            assistant_text("привет-котик"),
        ];
        let out = sanitize_messages(msgs);
        assert_eq!(out.len(), 2);
        assert_eq!(out.last().unwrap().role, Role::User);
    }

    #[test]
    fn keeps_assistant_with_tool_calls_at_end() {
        let msgs = vec![
            Message::system("s"),
            Message::user("привет"),
            assistant_with_call("сейчас минутку"),
            tool_result("Отправлено 1 сообщений"),
        ];
        let out = sanitize_messages(msgs);
        assert_eq!(out.len(), 4);
        assert_eq!(out.last().unwrap().role, Role::Tool);
    }

    #[test]
    fn drops_orphan_tool_message() {
        let msgs = vec![
            Message::system("s"),
            Message::user("привет"),
            tool_result("сирота"),
        ];
        let out = sanitize_messages(msgs);
        assert!(out.iter().all(|m| m.role != Role::Tool));
    }

    #[test]
    fn assistant_with_calls_without_answers_loses_calls() {
        let msgs = vec![
            Message::system("s"),
            Message::user("привет"),
            assistant_with_call("фото будет"),
        ];
        let out = sanitize_messages(msgs);
        let last = out.last().unwrap();
        // Вызовы сняты (ответов не было) — иначе API ждёт tool-результат.
        assert!(last.tool_calls.is_none());
    }

    #[test]
    fn drops_leading_tool_messages() {
        let msgs = vec![tool_result("в начале"), Message::system("s"), Message::user("привет")];
        let out = sanitize_messages(msgs);
        assert_eq!(out.first().unwrap().role, Role::System);
    }

    #[test]
    fn only_system_gets_stub_user() {
        let msgs = vec![Message::system("s")];
        let out = sanitize_messages(msgs);
        assert_eq!(out.last().unwrap().role, Role::User);
    }

    #[test]
    fn trailing_system_is_removed() {
        let msgs = vec![Message::user("привет"), Message::system("s")];
        let out = sanitize_messages(msgs);
        assert_eq!(out.last().unwrap().role, Role::User);
    }

    #[test]
    fn drops_all_trailing_assistants() {
        // Реальный кейс фазы 2: хвост истории кончается её репликой, а в
        // turn_ctx — assistant с неотвеченным вызовом (tool-результата нет).
        // Раньше снимался только один — и запрос падал с 400.
        let msgs = vec![
            Message::system("s"),
            Message::user("привет"),
            assistant_text("мой ответ"),
            assistant_with_call("фото будет"),
        ];
        let out = sanitize_messages(msgs);
        assert_eq!(out.last().unwrap().role, Role::User);
        assert!(!out.iter().any(|m| m.role == Role::Assistant));
    }

    #[test]
    fn normal_conversation_untouched() {
        let msgs = vec![
            Message::system("s"),
            Message::user("привет"),
            assistant_text("мурр"),
            Message::user("как дела"),
        ];
        let out = sanitize_messages(msgs);
        assert_eq!(out.len(), 4);
        assert_eq!(out.last().unwrap().role, Role::User);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_opt_none() {
        assert_eq!(parse_opt_f32("none"), None);
        assert_eq!(parse_opt_f32("0.9"), Some(0.9));
        assert_eq!(parse_opt_f32(""), None);
    }

    #[test]
    fn message_builders() {
        let m = Message::user("hi");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.content, MessageContent::Text("hi".to_string()));
        let t = Tool::new("open", "opens chat", serde_json::json!({}));
        assert_eq!(t.function.name, "open");
    }

    #[test]
    fn extract_clean_reply_strips_meta() {
        let raw = "- thought: думаю о чем-то\n- emotion: 🥰\n- to remember: факт\n- intention: ответить\n\n[Call tool: ask({\"q\"})]\n\nПривет, Саша! Как дела?";
        let clean = extract_clean_reply(raw).unwrap();
        assert!(!clean.contains("- thought:"));
        assert!(!clean.contains("[Call tool:"));
        assert!(clean.contains("Привет, Саша!"));
        assert!(clean.trim().starts_with("Привет"));
    }

    #[test]
    fn extract_thinking_native_trims() {
        // Новый extract_thinking просто тримирует native reasoning_content.
        assert_eq!(extract_thinking("  думаю о Саше  "), Some("думаю о Саше".into()));
        assert!(extract_thinking("   \n  ").is_none());
        assert!(extract_thinking("").is_none());
    }

    #[test]
    fn extract_thinking_from_content_fallback() {
        // Fallback парсит legacy `- thought:`/`- emotion:` строки из content.
        let raw = "- thought: думаю о Саше\n- emotion: 🥰\n- intention: ответить\nчистый текст ответа";
        let t = extract_thinking_from_content(raw).unwrap();
        assert!(t.contains("- thought: думаю о Саше"));
        assert!(t.contains("- emotion: 🥰"));
        assert!(!t.contains("чистый текст"));
        // Если таких строк нет — None.
        assert!(extract_thinking_from_content("просто текст ответа").is_none());
    }

    #[test]
    fn extract_clean_reply_empty_when_only_meta() {
        let raw = "- thought: x\n- emotion: 🥰\n[Call tool: ask({\"q\"})]";
        assert!(extract_clean_reply(raw).is_none());
    }

    #[test]
    fn extract_clean_reply_strips_reasoning_plain_content() {
        // Gemma иногда шлёт раздумье как plain content (не в reasoning_content).
        let raw = "💭 The user is asking a simple question: \"ты как?\". I will use #ask.\nThe user asked a direct question. My plan is to call #ask.\nTool selection: #ask";
        assert!(extract_clean_reply(raw).is_none());
    }

    #[test]
    fn extract_clean_reply_keeps_real_answer() {
        let raw = "💭 thinking here\nя нормально, Сашенька! а ты как? 🥰";
        let r = extract_clean_reply(raw).unwrap();
        assert!(r.contains("я нормально"));
        assert!(!r.contains("💭"));
    }

    #[test]
    fn parse_tool_calls_finds_ask() {
        let raw =
            "- thought: x\n[Call tool: ask({\"query\":\"что знаю о Саше?\"})]\n- intention: y";
        let calls = parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ask");
        assert!(calls[0].args_json.contains("что знаю о Саше"));
    }

    #[test]
    fn parse_tool_calls_nested_json() {
        let raw = "[Call tool: ask({\"a\":{\"b\":1},\"c\":2})]";
        let calls = parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ask");
        assert!(calls[0].args_json.contains("\"b\":1"));
    }

    #[test]
    fn parse_tool_calls_gemma_format() {
        let raw = "<|tool_call>call:ask{query:\"что знаю о Саше?\"}";
        let calls = parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ask");
        assert!(calls[0].args_json.contains("что знаю о Саше"));
    }

    #[test]
    fn parse_tool_calls_gemma_no_paren_with_tags() {
        // Реальный формат Gemma через OAI API: `ask{query:<|"|>...<|"|>}` (без
        // скобок, со внутренними тегами). До фикса parse не находил из-за
        // отсутствия '(' — этот тест ловит регрессию.
        let raw = "[Call tool: ask{query:<|\"|>Как я себя чувствую сейчас?<|\"|>}]\n- intention: ответить";
        let calls = parse_tool_calls(raw);
        assert_eq!(calls.len(), 1, "должен найти ask без скобок");
        assert_eq!(calls[0].name, "ask");
        // Теги <|"|> вырезаются, остаётся чистый JSON.
        assert!(!calls[0].args_json.contains("<|"));
        assert!(calls[0].args_json.contains("Как я себя чувствую"));
    }

    #[test]
    fn parse_tool_calls_none() {
        assert!(parse_tool_calls("просто текст без вызовов").is_empty());
    }

    // ── Порядок сообщений (анти-3230 гейт) ────────────────────────────────

    fn tool_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    #[test]
    fn assistant_tool_calls_serializes_tool_calls_array() {
        let m = Message::assistant_tool_calls("", vec![tool_call("c1", "send_telegram_message")]);
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "assistant");
        assert_eq!(v["content"], "");
        let tcs = v["tool_calls"].as_array().expect("tool_calls есть");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["type"], "function");
        assert_eq!(tcs[0]["id"], "c1");
        assert_eq!(tcs[0]["function"]["name"], "send_telegram_message");
    }

    #[test]
    fn tool_result_carries_binding_fields() {
        let m = Message::tool_result(Some("c1".into()), Some("ask".into()), "результат");
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "tool");
        assert_eq!(v["tool_call_id"], "c1");
        assert_eq!(v["name"], "ask");
        // Голый Message::tool связки не имеет — только для старого кода.
        let bare = serde_json::to_value(Message::tool("x")).unwrap();
        assert!(bare.get("tool_call_id").is_none());
    }

    #[test]
    fn tool_loop_order_is_mistral_valid() {
        // Валидная для Mistral последовательность агентного цикла (turn 2):
        // system → user → assistant(tool_calls) → tool(результат).
        // Раньше история собиралась как [system, tool, ...] → 400 3230.
        let messages = vec![
            Message::system("sys"),
            Message::user("привет"),
            Message::assistant_tool_calls("", vec![tool_call("c1", "ask")]),
            Message::tool_result(Some("c1".into()), Some("ask".into()), "найдено"),
        ];
        let v = serde_json::to_value(&messages).unwrap();
        let roles: Vec<&str> = v
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool"]);
        // tool-ответ идёт СРАЗУ после assistant с tool_calls (не после system).
        assert_eq!(v[2]["tool_calls"][0]["id"], v[3]["tool_call_id"]);
    }
}
