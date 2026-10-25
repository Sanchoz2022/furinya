//! Агент Telegram: цикл приёма update-ов, агентный loop ответа,
//! консолидация памяти при сне/idle, загрузка истории чата.
//!
//! Весь функционал общения с Telegram живёт здесь (а не в main.rs).

use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use furinya_core::config::Config;
use furinya_core::prompts::Prompts;
use furinya_diary::{
    Diary, Entry, SituationStore, active_context, consolidate_memory, read_active,
};
use std::collections::HashMap;
use furinya_llm::{
    ContentPart, LlmClient, Message, MessageContent, ToolCall, build_tools, extract_thinking,
    extract_thinking_from_content,
};
use furinya_vision::describe_photo;
use furinya_audio::{synthesize_speech_openai, synthesize_fish_voice, transcribe_audio};
use furinya_sd::generate_image;
use chrono::{Local, Timelike, Utc};
use rand::Rng;
use serde_json::Value;
use tracing::{debug, error, info, warn};

use crate::assessment::{
    Assessment, build_assess_input, format_assessment, parse_assessment, read_chat_state,
    write_chat_state,
};
use crate::{TdClient, UpdateLoop, auth, default_lib_path};

// ── Новости из Telegram-каналов (актуальность для разговоров) ─────────────
//
// Механика: аккаунт бота подписывается на новостные каналы (Россия + НН)
// и читает их через TDLib. Сообщения каналов — НЕ «разговор»: они кладутся
// в кэш, из которого берутся (а) заголовки в idle-стимул — чтобы ей было
// О ЧЁМ поговорить, (б) инструмент fresh_news для живого чата.

/// Одна новость из канала.
#[derive(Debug, Clone)]
struct NewsItem {
    chat_id: i64,
    chat_title: String,
    message_id: i64,
    /// Unix-время сообщения (поле date).
    ts: i64,
    text: String,
}

/// Кэш прочитанных новостей: поллер пишет, fresh_news и idle-стимул читают.
static NEWS: std::sync::LazyLock<std::sync::Mutex<Vec<NewsItem>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

/// Resolved новостные каналы: chat_id → заголовок канала.
static NEWS_CHANNELS: std::sync::LazyLock<std::sync::Mutex<HashMap<i64, String>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));

/// Максимальная ёмкость кэша (старьё вытирается).
const NEWS_CACHE_CAP: usize = 400;

fn push_news(item: NewsItem) {
    if item.text.chars().count() < 15 {
        // «Фото», пустые подписи и прочий мусор не нужны.
        return;
    }
    let mut v = NEWS.lock().unwrap();
    if v.iter()
        .any(|n| n.chat_id == item.chat_id && n.message_id == item.message_id)
    {
        return;
    }
    v.push(item);
    if v.len() > NEWS_CACHE_CAP {
        let excess = v.len() - NEWS_CACHE_CAP;
        v.drain(0..excess);
    }
}

/// Текст из TDLib-сообщения: messageText или caption у медиа.
fn news_message_text(msg: &Value) -> Option<String> {
    let content = &msg["content"];
    let t = content["text"]["text"]
        .as_str()
        .or_else(|| content["caption"]["text"].as_str())?
        .trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

/// Последние `n` новостей, отсортированные по свежести (новые раньше).
fn news_headlines(n: usize) -> Vec<NewsItem> {
    let mut window: Vec<NewsItem> = NEWS.lock().unwrap().iter().rev().take(200).cloned().collect();
    window.sort_by_key(|i| std::cmp::Reverse(i.ts));
    window.into_iter().take(n).collect()
}

/// Строка «— [канал · N мин назад]: текст» для отображения новости.
fn format_news_line(item: &NewsItem, max_chars: usize) -> String {
    let age_min = if item.ts > 0 {
        let elapsed = (Utc::now().timestamp() - item.ts) / 60;
        if elapsed <= 0 {
            "только что".to_string()
        } else if elapsed < 60 {
            format!("{} мин назад", elapsed)
        } else {
            format!("{} ч назад", elapsed / 60)
        }
    } else {
        "недавно".to_string()
    };
    let mut t: String = item.text.chars().take(max_chars).collect();
    if item.text.chars().count() > max_chars {
        t.push('…');
    }
    let t = t.replace('\n', " ");
    format!("— [{} · {}]: {}", item.chat_title, age_min, t)
}

/// Блок заголовков для инъекции в контекст (idle-стимул). Пусто, когда новостей нет.
fn news_hint(max_items: usize, max_chars_each: usize) -> String {
    let items = news_headlines(max_items);
    if items.is_empty() {
        return String::new();
    }
    let mut s = String::from(
        "\n[свежее из новостных каналов — если захочешь первой написать о чем-то актуальном, можешь коротко упомянуть и добавить СВОЁ мнение; не пересказывай длинно]\n",
    );
    for it in &items {
        s.push_str(&format_news_line(it, max_chars_each));
        s.push('\n');
    }
    s
}

// ── Фото: промпт → картинка ───────────────────────────────────────────────
//
// Проблемы, которые закрываем:
//  1) LLM пишет сцену ПО-РУССКИ → SD1.5/SDXL русский не понимает → на выходе
//     generic-портрет в каждом кадре («однотипно»). Лечится переводом.
//  2) Канон внешности подставляется перед сценой, но модель часто дублирует
//     его сама → промпт раздувается и веса тегов едут. Лечится дедупом.
//  3) Модель рисует одну и ту же сцену → нужен принудительный разброс.

/// Есть ли кириллица в строке (признак русского промпта).
// ── Stable Diffusion prompt engineering ──────────────────────────────────
//
// Порт kuni `ImageGenerator::engineerPrompt`: отдельный LLM-вызов превращает
// свободное описание сцены в короткий danbooru-промпт на базе канона.
// Раньше здесь стояли самодельные фильтры (strip_eye_tags и т.п.), которые
// вырезали из сцены ЛЮБОЙ цвет («blue dress», «red rose», «golden sunlight»)
// — из-за этого все фото выходили однотипными. Дизайн персонажа теперь держит
// prompt-инженер, а сцена не теряет деталей.

/// Явные NSFW-слова: обычной LLM их не показываем (может отказаться), но
/// возвращаем в позитивный промпт после инжиниринга (трюк из kuni).
const SD_NSFW_WORDS: &[&str] = &[
    "explicit nudity",
    "explicit nude",
    "explicit erotic",
    "pussy",
    "breasts",
    "nsfw",
    "genital",
    "vagina",
    "penis",
    "nude",
    "naked",
    "topless",
    "nipples",
    "areola",
    "erotic",
];

/// Дефолтный негативный промпт для локального SD1.5.
const DEFAULT_SD_NEGATIVE: &str = "worst quality, low quality, jpeg artifacts, watermark, \
    text, signature, blurry, deformed, mutated, bad anatomy, bad hands, poorly drawn hands, \
    missing fingers, fused fingers, extra fingers, extra hands, extra arms, extra limbs, \
    malformed limbs, disconnected limbs, extra tail, multiple tails, two tails, \
    ugly, both eyes same color, matching eyes, symmetric eyes, cross-eyed, extra eyes, \
    dull eyes, large breasts, huge breasts, thick thighs, tall, same pose, same background, \
    cropped head";

/// Регистронезависимая замена подстроки (для вырезания NSFW-слов).
fn replace_case_insensitive(haystack: &str, needle: &str, with: &str) -> String {
    if needle.is_empty() {
        return haystack.to_string();
    }
    let lower = haystack.to_lowercase();
    let n = needle.to_lowercase();
    let mut out = String::with_capacity(haystack.len());
    let mut start = 0;
    while let Some(pos) = lower[start..].find(&n) {
        let abs = start + pos;
        out.push_str(&haystack[start..abs]);
        out.push_str(with);
        start = abs + n.len();
    }
    out.push_str(&haystack[start..]);
    out
}

/// Вырезает NSFW-слова из описания перед отправкой prompt-инженеру.
fn sanitize_for_llm(scene: &str) -> String {
    let mut out = scene.to_string();
    for w in SD_NSFW_WORDS {
        out = replace_case_insensitive(&out, w, "");
    }
    // Чистим осиротевшие запятые/двойные пробелы.
    let mut cleaned = String::with_capacity(out.len());
    for ch in out.chars() {
        if ch == ',' {
            if !cleaned.ends_with(',') && !cleaned.trim_end().is_empty() {
                cleaned.push(ch);
            }
        } else if ch.is_whitespace() {
            if !cleaned.ends_with(' ') {
                cleaned.push(' ');
            }
        } else {
            cleaned.push(ch);
        }
    }
    cleaned
        .trim_matches(|c: char| c == ',' || c.is_whitespace())
        .to_string()
}

/// NSFW-слова, реально встречавшиеся в описании (возвращаем в промпт).
fn explicit_words_in(scene: &str) -> Vec<&'static str> {
    let lower = scene.to_lowercase();
    SD_NSFW_WORDS
        .iter()
        .copied()
        .filter(|w| lower.contains(w))
        .collect()
}

/// Парсит JSON от prompt-инженера: допускает ```json ... ``` и лишний текст.
fn parse_sd_json(content: &str) -> anyhow::Result<(String, String)> {
    let mut s = content.trim();
    if let Some(start) = s.find("```json") {
        s = &s[start + 7..];
        if let Some(end) = s.find("```") {
            s = &s[..end];
        }
    } else if let Some(start) = s.find("```") {
        s = &s[start + 3..];
        if let Some(end) = s.find("```") {
            s = &s[..end];
        }
    }
    let start = s
        .find('{')
        .ok_or_else(|| anyhow::anyhow!("нет JSON-объекта в ответе prompt-инженера"))?;
    let end = s
        .rfind('}')
        .ok_or_else(|| anyhow::anyhow!("нет конца JSON-объекта в ответе prompt-инженера"))?;
    let v: Value = serde_json::from_str(&s[start..=end])?;
    let positive = v
        .get("positivePrompt")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let negative = v
        .get("negativePrompt")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if positive.is_empty() {
        anyhow::bail!("prompt-инженер вернул пустой positivePrompt");
    }
    Ok((positive, negative))
}

/// Обрезает промпт по тегам до `max_words` слов, не разрывая тег.
fn clamp_prompt_words(prompt: &str, max_words: usize) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut words = 0usize;
    for tag in prompt.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let n = tag.split_whitespace().count();
        if words + n > max_words && !out.is_empty() {
            break;
        }
        out.push(tag);
        words += n;
    }
    out.join(", ")
}

/// Достаёт текст из ответа LLM (content — enum Text | Parts).
fn response_text(resp: &furinya_llm::ChatResponse) -> String {
    let raw = resp
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_else(|| MessageContent::Text(String::new()));
    match raw {
        MessageContent::Text(t) => t,
        MessageContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// Канон для SD: заменяет обычные теги глаз на усиленную гетерохромию.
/// Так глаза не «плывут» даже на широких/боковых кадрах.
fn strengthen_eyes(canon: &str) -> String {
    let kept: Vec<&str> = canon
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .filter(|t| {
            let low = t.to_lowercase();
            !low.contains("eye") && !low.contains("heterochromia")
        })
        .collect();
    format!(
        "{}, (heterochromia:1.5), heterochromatic eyes, one purple eye, one blue eye",
        kept.join(", ")
    )
}

/// Убирает точные дубли тегов (безопасно, в отличие от подстрочного дедупа).
fn dedupe_exact_tags(prompt: &str) -> String {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for tag in prompt.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if seen.insert(tag.to_lowercase()) {
            out.push(tag);
        }
    }
    out.join(", ")
}

/// Порт kuni `engineerPrompt`, но с разделением труда: канон персонажа
/// (тело, волосы, уши, хвост, глаза) собирается в Rust и всегда идёт первым,
/// а LLM добавляет ТОЛЬКО теги сцены: действие, поза, ракурс, место, одежда,
/// свет, настроение. Так модель не может потерять канон или заменить глаза.
async fn engineer_sd_prompt(
    llm: &LlmClient,
    scene: &str,
    nsfw: bool,
) -> anyhow::Result<(String, String)> {
    let cool = llm.with_temperature(0.5);
    let nsfw_hint = if nsfw {
        "The scene may imply adult intimacy, but do NOT write explicit sexual terms."
    } else {
        ""
    };
    let system = format!(
        r#"You are an expert Stable Diffusion prompt engineer for an anime SD1.5 model.
The character's appearance (hair, cat ears, tail, heterochromia, body, face) is HARD-CODED
and will be prepended automatically. Your job is ONLY the scene around her.

POSITIVE PROMPT RULES:
- English only. Comma-separated danbooru-style tags/keywords, NEVER full sentences.
- Do NOT describe the character's hair, ears, tail, eyes, body, face or skin color.
- Output ONLY scene tags: action/pose, camera angle, location, outfit, accessories,
  lighting, time of day, mood, composition.
- Preserve EVERY element of the desired photo: action, pose, held objects, location,
  time of day, mood. Never drop the action or the objects.
- Always include at least one concrete outfit tag that fits the scene (e.g. red knitted
  sweater, black dress, oversized shirt, bikini). Vary outfits between scenes.
- If the description contains action/motion (dancing, running, jumping, stretching,
  fighting, playing), make the shot DYNAMIC: action shot, dynamic pose, motion,
  flowing hair, flowing clothes, dynamic camera angle. Never answer an action scene
  with a static standing pose.
- If the description is a calm portrait/selfie, a calm pose is fine.
- For close-up/portrait/selfie compositions, include "both eyes visible" so the
  heterochromia reads clearly. Do not add it for profile/side views.
- Include "selfie" or "photo" if it fits (most requests are selfies).
- Do NOT request text, signs, posters or written words.
- Keep it compact: at most 25 words. The full prompt (with the hard-coded character)
  must stay under 60 words.
- Vary composition: choose a fresh camera angle instead of a default front view.

NEGATIVE PROMPT RULES:
- Comma-separated tags: worst quality, low quality, blurry, jpeg artifacts, watermark,
  signature, text, bad anatomy, bad hands, poorly drawn hands, missing fingers,
  fused fingers, extra fingers, extra hands, extra arms, extra limbs, malformed limbs,
  disconnected limbs, deformed, mutated, extra tail, multiple tails, two tails,
  cropped, out of frame.
- Always include these tags: both eyes same color, matching eyes.
- Do NOT include character design tags in the negative prompt.

{nsfw_hint}

OUTPUT: strict JSON only, no markdown fences:
{{"positivePrompt": "...", "negativePrompt": "..."}}"#,
        nsfw_hint = nsfw_hint,
    );
    let messages = vec![
        Message::system(system),
        Message::user(format!("Desired photo: {scene}")),
    ];
    let resp = cool.chat(messages, None).await?;
    let content = response_text(&resp);
    let (positive, negative) = parse_sd_json(&content)?;
    Ok((clamp_prompt_words(&positive, 25), negative))
}

/// Возвращает в готовый промпт NSFW-термины, вырезанные перед инженером.
fn append_explicit_words(positive: &str, original_scene: &str) -> String {
    let mut tags: Vec<String> = positive
        .split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    for w in explicit_words_in(original_scene) {
        if !tags.iter().any(|t| t.eq_ignore_ascii_case(w)) {
            tags.push(w.to_string());
        }
    }
    tags.join(", ")
}

/// Вырезает Gemma-теги `<|...|>` из текста (до парсинга JSON-аргументов).
#[allow(clippy::collapsible_if)]
fn strip_gemma_tags(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
            if let Some(end) = s[i + 2..].find("|>") {
                i = i + 2 + end + 2;
                continue;
            }
        }
        let ch_len = match bytes[i] {
            b if b < 0x80 => 1,
            b if b >> 5 == 0b110 => 2,
            b if b >> 4 == 0b1110 => 3,
            b if b >> 3 == 0b11110 => 4,
            _ => 1,
        };
        if i + ch_len <= bytes.len() {
            out.push_str(&s[i..i + ch_len]);
        }
        i += ch_len;
    }
    out
}

/// Вырезает служебные строки раздумья (`- thought:`, `thought:`, `emotion:`,
/// `to remember:`, `intention:`, `tool:` и т.п.) из текста. Используется как
/// последняя линия защиты: даже если модель засунет раздумье внутрь `parts`
/// инструмента, оно не улетит в чат.
fn strip_meta_lines(s: &str) -> String {
    let meta: &[&str] = &[
        "- thought:",
        "thought:",
        "- emotion:",
        "emotion:",
        "- to remember:",
        "to remember:",
        "- intention:",
        "intention:",
        "- tool:",
        "tool:",
    ];
    let mut out = String::with_capacity(s.len());
    for line in s.lines() {
        let t = line.trim();
        if meta.iter().any(|p| t == *p || t.starts_with(p)) {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim().to_string()
}

/// Вырезает ролевые действия в звёздочках (`*поднимает бровь*`) — для TTS:
/// озвучивать описания меню нельзя, Statue должна произносить только живую
/// речь (спецификация §4.2: r"\*[^*]+\*" → пусто).
fn strip_asterisk_actions(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('*') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        match after.find('*') {
            Some(j) => rest = &after[j + 1..], // парная пара — дропаем действие
            None => {
                // Одиночная звезда — не действие; не душим текст.
                rest = after;
                break;
            }
        }
    }
    out.push_str(rest);
    // Схлопывание лишних пробелов, оставшихся от удалённых действий.
    let mut collapsed = String::with_capacity(out.len());
    let mut prev_ws = false;
    for c in out.chars() {
        let ws = c.is_whitespace();
        if !(ws && prev_ws) {
            collapsed.push(c);
        }
        prev_ws = ws;
    }
    collapsed.trim().to_string()
}

/// Выкусывает поле `query` из JSON-аргументов tool-call.
fn extract_query_arg(args: &str) -> Option<String> {
    let v: Value = serde_json::from_str(args).ok()?;
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

/// Форматирует unix-время в локальный штамп `[ДД.ММ ЧЧ:ММ]`.
/// Если `ts == 0` (неизвестно) — возвращает пустую строку.
fn format_ts(ts: i64) -> String {
    if ts <= 0 {
        return String::new();
    }
    use std::time::UNIX_EPOCH;
    let secs = ts as u64;
    match UNIX_EPOCH.checked_add(std::time::Duration::from_secs(secs)) {
        Some(t) => {
            let dt: chrono::DateTime<chrono::Local> = t.into();
            format!("[{}]", dt.format("%d.%m %H:%M"))
        }
        None => String::new(),
    }
}

/// Вставляет сообщение в живую историю чата, подрезая старый хвост
/// (кроме system[0]): длинная сессия не должна расти бесконечно.
fn remember_in_history(hist: &mut Vec<Message>, m: Message) {
    hist.push(m);
    while hist.len() > 200 {
        hist.remove(1);
    }
}

/// Сохраняет «что я отправила» в живую историю (assistant) и в полную
/// историю (tg_history_full, (sender, is_me=true, text, date)) — раньше
/// собственные ответы бота не запоминались вовсе («не работает память»).
fn record_my_send(
    tg_history: &mut Vec<Message>,
    tg_history_full: &mut Vec<(i64, bool, String, i64)>,
    my_id: i64,
    cfg: &Config,
    parts: &[String],
) {
    if parts.is_empty() {
        return;
    }
    let now_unix = Utc::now().timestamp();
    let joined = parts.join("\n\n");
    remember_in_history(
        tg_history,
        Message::assistant(format!(
            "[{}] {}: {}",
            format_ts(now_unix),
            cfg.general.character_name,
            joined
        )),
    );
    for p in parts {
        tg_history_full.push((my_id, true, p.clone(), now_unix));
    }
}

/// Последние собственные реплики персонажа — подсказка «не повторяйся»
/// для idle/сон-стимулов (без неё модель повторяет одну строчку).
fn own_recent_quotes(
    history: &[(i64, bool, String, i64)],
    n: usize,
) -> String {
    let own: Vec<String> = history
        .iter()
        .filter(|(_, is_me, t, _)| *is_me && !t.trim().is_empty())
        .rev()
        .take(n)
        .map(|(_, _, t, _)| t.chars().take(90).collect::<String>())
        .collect();
    if own.is_empty() {
        return String::new();
    }
    format!(
        " Твои последние реплики: {}. НЕ повторяй их и не пиши снова «я тут, жду тебя» — \
         это уже было. Скажи что-то НОВОЕ — или промолчи.",
        own.iter()
            .map(|s| format!("«{s}»"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Обрезает строку по символам (без разрыва UTF-8).
fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Создаёт `self_state.md` при старте, если его ещё нет.
///
/// Зачем: idle-шанс (5–15 мин) наступает РАНЬШЕ, чем консолидация-сон (15 мин),
/// поэтому без бутстрапа первый шанс «написать первой» приходил бы с пустым
/// состоянием — и модель молчала бы. Строим состояние из уже загруженной
/// истории (тот же формат `[ДД.ММ ЧЧ:ММ] Имя: текст`, что и в консолидации).
/// Идемпотентно: если файл есть — сразу выходим.
async fn bootstrap_self_state(
    llm: &LlmClient,
    prompts: &Prompts,
    cfg: &Config,
    tg_history_full: &[(i64, bool, String, i64)],
) {
    let path = std::path::Path::new(&cfg.misc.diary_dir).join("self_state.md");
    if path.exists() {
        return;
    }
    let mut chat_text = String::new();
    for (sender, is_me, text, date) in tg_history_full {
        if text.trim().is_empty() {
            continue;
        }
        let who = if *is_me {
            cfg.general.character_name.clone()
        } else if *sender == cfg.general.papik_chat_id {
            cfg.general.papik_name.clone()
        } else {
            format!("user{}", sender)
        };
        chat_text.push_str(&format!("[{}] {}: {}\n", format_ts(*date), who, text));
    }
    // Температура как у консолидации (0.2): иначе модель уходит в
    // мелодраматичный дрейф и нарушает запреты self_state-промпта.
    // Логика генерации + guard от «нужды» — общая с консолидацией.
    let cool = llm.with_temperature(0.2);
    furinya_diary::consolidate::update_self_state(&cool, prompts, &cfg.misc.diary_dir, &chat_text)
        .await;
    if path.exists() {
        info!("[tg] self_state создан при старте: {:?}", path);
    }
}

/// Собирает блок памяти в рамках бюджета символов.
/// Без этого память (дневник + выжимка чата) раздувала контекст до 100+ KB
/// и упиралась в окно модели.
fn budgeted_block(parts: &[String], budget: usize) -> String {
    let mut out = String::new();
    for p in parts {
        if out.chars().count() >= budget {
            break;
        }
        let rest = budget - out.chars().count();
        let piece = take_chars(p, rest);
        let full = piece.chars().count() == p.chars().count();
        out.push_str(&piece);
        if !full {
            out.push_str("…");
            break;
        }
    }
    out
}

/// Извлекает факты о собеседнике из per-чат выжимок в векторную память (RAG).
///
/// Без этого шага дневник остаётся пустым: консолидация только ПЕРЕЖИМАЕТ
/// старые записи, а новых знаний не создаёт. Факты берутся из файла
/// `data/diary/chat_history/<chat_id>.md`, который обновляет консолидация.
async fn extract_facts_to_diary(
    llm: &LlmClient,
    diary: &Diary,
    prompts: &Prompts,
    diary_entries: &mut Vec<furinya_diary::Entry>,
    cfg: &Config,
) {
    // Извлечение фактов — служебное: холодная температура (точность важнее,
    // живой 0.8 тут не нужен).
    let cool_llm = llm.with_temperature(0.2);
    match furinya_diary::build_profile_memory(
        &cool_llm,
        diary,
        prompts,
        &cfg.misc.diary_dir,
        diary_entries,
        cfg.misc.diary_keep_last,
        cfg.misc.active_top_k,
    )
    .await
    {
        Ok(n) => {
            *diary_entries = diary.load_all().unwrap_or_default();
            info!("[tg] факты в память (RAG): {} записей", n);
        }
        Err(e) => error!("[tg] извлечение фактов ошибка: {}", e),
    }
}

/// Состояние анти-повтора агента: кэш эмбеддингов своих реплик (текст → вектор)
/// и «индульгенция» kuni (+0.07 за каждый подряд срабатывающий ход).
struct AntiRepeatState {
    emb_cache: HashMap<String, Vec<f32>>,
    head_start: f32,
    /// Черновики, заблокированные анти-повтором В ЭТОМ ходу. Заблокированный
    /// текст НЕ уходит в чат → он не попадает ни в историю, ни в `seen`, и
    /// модель следующим вызовом отправляла его же (баг 23.10: блок «Вот так я
    /// танцую…», а через 5 секунд тот же текст ушёл в чат). Держим их здесь и
    /// сравниваем с ними следующие черновики.
    blocked: Vec<String>,
}

impl AntiRepeatState {
    fn new() -> Self {
        Self {
            emb_cache: HashMap::new(),
            head_start: 0.0,
            blocked: Vec::new(),
        }
    }

    /// Запоминает заблокированный черновик (кольцевой буфер, 32 штуки).
    fn remember_blocked(&mut self, text: &str) {
        let t = text.trim();
        if t.is_empty() {
            return;
        }
        self.blocked.push(t.to_string());
        if self.blocked.len() > 32 {
            let cut = self.blocked.len() - 32;
            self.blocked.drain(..cut);
        }
    }
}

/// Порог лексической (символьные 3-граммы) похожести. Подобран на реальных
/// репликах бота: дословные самоповторы дают 0.73–1.0, нормально разные —
/// меньше 0.51 (даже с одинаковым обращением «моя кошечка» — 0.50).
const LEXICAL_REPEAT_THRESHOLD: f32 = 0.68;

/// Порог для ЧАСТЕЙ ОДНОГО ответа: part2 не должен быть парафразом part1.
/// Строже межходового: две «разные» реплики в одной пачке очень часто несут
/// одну мысль («я не поняла, что ты ждёшь…» + «я не поняла, что ты ждёшь…»),
/// и лексика ловит это раньше, чем эмбеддинги.
const PARTS_REPEAT_THRESHOLD: f32 = 0.55;

/// Косинус символьных n-грамм (n=3) — локальная страховка от дословных
/// самоповторов, работающая БЕЗ сети.
///
/// Mistral embeddings регулярно отвечают `429 backend_out_of_capacity`, и
/// тогда эмбеддинг-анти-повтор молча пропускал копипасту собственных реплик
/// («Ты — моя единственная ласка…» из хода в ход). Лексика ловит именно
/// повторы, не задевая нормально разные ответы.
fn lexical_similarity(a: &str, b: &str) -> f32 {
    use std::collections::HashMap;
    const N: usize = 3;

    /// Нижний регистр, всё не-буквенно-цифровое → одиночный пробел,
    /// пробелы схлопнуты, края обрезаны (как в калибровочной модели).
    fn norm(s: &str) -> Vec<char> {
        let mut out: Vec<char> = Vec::with_capacity(s.len());
        for c in s.chars().flat_map(|c| c.to_lowercase()) {
            let c = if c.is_alphanumeric() { c } else { ' ' };
            if c == ' ' && (out.is_empty() || out.last() == Some(&' ')) {
                continue;
            }
            out.push(c);
        }
        while out.last() == Some(&' ') {
            out.pop();
        }
        out
    }

    fn grams(chars: &[char]) -> HashMap<Vec<char>, f64> {
        let mut m: HashMap<Vec<char>, f64> = HashMap::new();
        if chars.is_empty() {
            return m;
        }
        if chars.len() < N {
            *m.entry(chars.to_vec()).or_insert(0.0) += 1.0;
            return m;
        }
        for w in chars.windows(N) {
            *m.entry(w.to_vec()).or_insert(0.0) += 1.0;
        }
        m
    }

    let ga = grams(&norm(a));
    let gb = grams(&norm(b));
    if ga.is_empty() || gb.is_empty() {
        return 0.0;
    }
    let dot: f64 = ga
        .iter()
        .map(|(k, va)| va * gb.get(k).copied().unwrap_or(0.0))
        .sum();
    let na: f64 = ga.values().map(|v| v * v).sum::<f64>().sqrt();
    let nb: f64 = gb.values().map(|v| v * v).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)) as f32
}

/// Дежурные формулы-пустышки, которые модель выдаёт вместо реакции на его
/// конкретные слова. Возвращает сработавший шаблон — чтобы tool-ошибка
/// объяснила модели, ЧТО именно не так.
///
/// Это не «запрет на слова», а защита от готовых утешительных конструкций:
/// «я тут / я рядом», «хочешь, чтобы я…», «расскажи ещё», «застряла в
/// мыслях», «иногда тишина говорит больше». Внутри содержательной фразы
/// («я тут подумала…») они допустимы — поэтому голые предложения-пустышки
/// режем отдельно, по предложению.
fn cliche_hit(text: &str) -> Option<&'static str> {
    let low = text.to_lowercase().replace('ё', "е");

    const PHRASES: &[(&str, &str)] = &[
        ("я тут, ", "голая формула присутствия («я тут / я рядом»)"),
        ("я здесь, ", "голая формула присутствия («я тут / я рядом»)"),
        ("я рядом, ", "голая формула присутствия («я тут / я рядом»)"),
        ("может, просто скажи", "дежурное «может, просто скажи…»"),
        ("может просто скажи", "дежурное «может, просто скажи…»"),
        ("я всегда рядом", "формула «я всегда рядом»"),
        ("я буду рядом", "формула «я буду рядом»"),
        ("ты ведь знаешь", "заготовка «ты ведь знаешь…»"),
        ("ты не один", "утешительная формула «ты не одинок»"),
        ("хочешь, чтобы я", "вопрос-формула «хочешь, чтобы я…»"),
        ("хочешь чтобы я", "вопрос-формула «хочешь, чтобы я…»"),
        ("мне кажется, ты", "догадка-формула «мне кажется, ты…»"),
        ("мне кажется ты", "догадка-формула «мне кажется, ты…»"),
        ("как помочь", "ассистентская формула «как помочь»"),
        ("чем помочь", "ассистентская формула «чем помочь»"),
        ("расскажи еще", "дежурное «расскажи ещё»"),
        ("застряла в своих мыслях", "отговорка «застряла в своих мыслях»"),
        ("застряла в мыслях", "отговорка «застряла в мыслях»"),
        ("зависла в мыслях", "отговорка «зависла в мыслях»"),
        ("запуталась в своих мыслях", "отговорка «запуталась в мыслях»"),
        ("запуталась в мыслях", "отговорка «запуталась в мыслях»"),
        ("давай я спрошу", "заготовка «давай я спрошу…»"),
        ("иногда даже тишина", "красивость «иногда даже тишина…»"),
        ("тишина говорит больше", "красивость «тишина говорит больше»"),
        ("молчу, потому что", "заготовка «молчу, потому что…»"),
        ("молчу потому что", "заготовка «молчу, потому что…»"),
        ("просто была рядом", "формула «просто была рядом»"),
        ("просто буду рядом", "формула «просто буду рядом»"),
        ("просто сидела рядом", "формула «просто сидела рядом»"),
        ("ты говоришь о чем-то, что важно", "заготовка «ты говоришь о чём-то важном»"),
        ("о чем-то, что важно для тебя", "заготовка «о чём-то важном для тебя»"),
    ];
    for (p, why) in PHRASES {
        if low.contains(p) {
            return Some(why);
        }
    }

    // Голое предложение-присутствие: «я тут», «я здесь», «я рядом»,
    // «я с тобой» и их склейки. Внутри фразы («я тут подумала…») — можно.
    for sentence in low.split(['.', '!', '?', '…', '\n']) {
        let s: String = sentence
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == ' ')
            .collect();
        let mut s = s.split_whitespace().collect::<Vec<_>>().join(" ");
        // Срезаем обращение и вводные союзы, чтобы «Но я здесь.» тоже
        // распознавалось как голая формула, а не как содержательная фраза.
        loop {
            let mut cut = false;
            for prefix in [
                "сашенька ",
                "александр ",
                "саша ",
                "саш ",
                "но ",
                "и ",
                "а ",
                "просто ",
            ] {
                if let Some(rest) = s.strip_prefix(prefix) {
                    s = rest.trim_start().to_string();
                    cut = true;
                    break;
                }
            }
            if !cut {
                break;
            }
        }
        let s = s.trim_matches([',', ' ']);
        if matches!(
            s,
            "я тут"
                | "я здесь"
                | "я рядом"
                | "я с тобой"
                | "я тут рядом"
                | "я рядом с тобой"
                | "я здесь рядом"
                | "я тут с тобой"
                | "я не ушла"
                | "я не уйду"
                | "я никуда не ушла"
                | "я никуда не уйду"
                | "я не бросила"
                | "я все еще здесь"
                | "я все еще тут"
        ) {
            return Some("голая формула присутствия («я тут / я рядом»)");
        }
    }
    None
}

/// Обрубок вместо реплики: одно слово («Сашенька…») или недописанная мысль
/// («Ты просто так спросили…»). Такую часть не отправляем: это пинг внимания,
/// а не ответ. Живую короткую реплику («Я не знаю.») не трогаем.
fn fragment_hit(text: &str) -> Option<&'static str> {
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        return Some("пустая часть");
    }
    if words.len() == 1 {
        return Some("обрубок из одного слова");
    }
    let t = text.trim_end();
    let t = t.trim_start();
    let starts_dots = t.starts_with("...") || t.starts_with('…');
    let ends_dots = t.ends_with("...") || t.ends_with('…');
    if starts_dots {
        // «...ну что тут думать-то?» — обрывок продолжения, которого в чате
        // не было: сообщение начинается с многоточия и читается как мусор.
        return Some("часть начинается с многоточия (обрывок мысли)");
    }
    if words.len() <= 6 && ends_dots {
        return Some("недописанная мысль с многоточием");
    }
    None
}

/// Эмбеддинг-защита от самоповторов — порт kuni (send_telegram_message.cpp,
/// «LLM is repeating itself»). Проверяет черновик ответа против недавних
/// реплик бота: если максимальная похожесть выше anti_repeat_trigger_max
/// (0.95) или средняя выше anti_repeat_trigger_avg (0.85) — отправка
/// отменяется, модель передумывает (переформулирует или молчит).
///
/// kuni-комментарий: после введения этой проверки качество диалогов
/// выросло заметно — модель перестаёт копипастить саму себя и либо
/// переключается на другую тему, либо честно молчит.
///
/// Ошибка эмбеддинга НЕ блокирует отправку (мягкая деградация).
async fn anti_repeat_check(
    candidate: &str,
    own_recent: &[String],
    diary: &Diary,
    cfg: &Config,
    st: &mut AntiRepeatState,
) -> Result<(), String> {
    let cand = candidate.trim();
    if cand.is_empty()
        || own_recent.is_empty()
        || cfg.misc.anti_repeat_trigger_max >= 1.0
        || cfg.misc.anti_repeat_trigger_avg >= 1.0
    {
        return Ok(());
    }
    // ЛОКАЛЬНАЯ проверка (без сети) — первым делом: эмбеддинги Mistral
    // постоянно лежат (429 backend_out_of_capacity), и без неё дословные
    // самоповторы уходили в чат. Порог подобран так, что нормально разные
    // реплики его не задевают.
    let mut lex_max = 0.0f32;
    let mut lex_quote = "";
    for own in own_recent {
        let sim = lexical_similarity(cand, own);
        if sim > lex_max {
            lex_max = sim;
            lex_quote = own;
        }
    }
    if lex_max > LEXICAL_REPEAT_THRESHOLD {
        debug!(
            "[tg] анти-повтор (лексика): {lex_max:.2} — черновик {:?} ≈ своё {:?}",
            cand,
            lex_quote.chars().take(60).collect::<String>()
        );
        return Err(format!("lexicalSimilarity={lex_max:.2}"));
    }
    // Чего не хватает в кэше — берём одной пачкой (Mistral: 1 RPS,
    // один batch-запрос вместо десятка одиночных).
    let mut need = vec![cand.to_string()];
    for own in own_recent {
        if !st.emb_cache.contains_key(own) && !need.contains(own) {
            need.push(own.clone());
        }
    }
    if let Ok(vecs) = diary.embedding().embed_batch(&need).await {
        for (t, v) in need.into_iter().zip(vecs) {
            st.emb_cache.insert(t, v);
        }
    } else {
        // Кэш не помог: пробуем хотя бы черновик по одному.
        match diary.embedding().embed(cand).await {
            Ok(v) => {
                st.emb_cache.insert(cand.to_string(), v);
            }
            Err(e) => {
                tracing::warn!("[tg] анти-повтор: эмбеддинги недоступны ({e}) — пропускаю");
                return Ok(());
            }
        }
    }
    let Some(target) = st.emb_cache.get(cand).cloned() else {
        return Ok(());
    };
    let mut max_sim = 0.0f32;
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for own in own_recent {
        let Some(emb) = st.emb_cache.get(own) else {
            continue;
        };
        let sim = furinya_diary::cosine_similarity(&target, emb);
        sum += sim;
        count += 1;
        max_sim = max_sim.max(sim);
    }
    if count == 0 {
        return Ok(());
    }
    let avg = sum / count as f32;
    debug!(
        "[tg] анти-повтор: max={max_sim:.2} avg={avg:.2} (порог {}/{} + {:.2})",
        cfg.misc.anti_repeat_trigger_max, cfg.misc.anti_repeat_trigger_avg, st.head_start
    );
    if max_sim > cfg.misc.anti_repeat_trigger_max + st.head_start {
        st.head_start += 0.07; // индульгенция kuni: повторяемости прощают больше
        return Err(format!("maxSimilarity={max_sim:.2}"));
    }
    if avg > cfg.misc.anti_repeat_trigger_avg + st.head_start {
        st.head_start += 0.07;
        return Err(format!("avgSimilarity={avg:.2}"));
    }
    st.head_start = 0.0; // успех — индульгенция сбрасывается
    // Ротация кэша (как в kuni: разросся — чистим весь).
    if st.emb_cache.len() >= cfg.misc.anti_repeat_max_history.max(2) {
        st.emb_cache.clear();
    }
    Ok(())
}

/// Исполняет TG-инструмент (send_telegram_message / react_with_emoji / ask / ...).
/// Возвращает `(результат, отправленное)`:
/// - результат — текст для tool-ответа модели;
/// - отправленное — что ФАКТИЧЕСКИ ушло в чат (части текста, [голосовое: …],
///   [фото: …]) — для записи в живую память (record_my_send).
#[allow(clippy::ptr_arg)]
#[allow(clippy::too_many_arguments)]
async fn execute_tg_tool(
    name: &str,
    args: &str,
    client: &TdClient,
    chat_id: i64,
    message_id: i64,
    llm: &LlmClient,
    diary: &Diary,
    diary_entries: &mut Vec<Entry>,
    situation: &SituationStore,
    // Сколько ещё сообщений разрешено отправить в этом ходу
    // (`max_message_parts - parts_sent`). send_telegram_message не может
    // превысить остаток — иначе вторым вызовом модель досылала ещё пачку.
    parts_remaining: usize,
    cfg: &Config,
    own_recent: &[String],
    anti_repeat_prompt: &str,
    // Его последнее сообщение — чтобы tool-ошибка про клише могла ткнуть
    // модель носом в конкретные слова, на которые надо отвечать.
    his_last: &str,
    rep: &mut AntiRepeatState,
) -> (String, Vec<String>) {
    match name {
        "send_telegram_message" => {
            let cleaned_args = strip_gemma_tags(args);
            let v: Value = match serde_json::from_str(&cleaned_args) {
                Ok(v) => v,
                Err(e) => {
                    return (
                        format!("ОШИБКА: невалидный JSON в send_telegram_message: {e}"),
                        Vec::new(),
                    )
                }
            };
            let parts = v
                .get("parts")
                .and_then(|p| p.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(strip_gemma_tags))
                        // Вырезаем служебные строки раздумья, если модель
                        // всё же засунула их внутрь parts (защита от утечки).
                        .map(|s| strip_meta_lines(&s))
                        .filter(|s| !s.trim().is_empty())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if parts.is_empty() {
                return (
                    "ОШИБКА: send_telegram_message получил пустой parts — нечего отправлять."
                        .to_string(),
                    Vec::new(),
                );
            }
            // Жёсткий потолок за ХОД (не за вызов): остаток бюджета приходит
            // снаружи. Иначе модель первым вызовом слала 2, вторым — ещё 3.
            if parts_remaining == 0 {
                return (
                    "Лимит сообщений на этот ход исчерпан — больше не отправляй, \
                     коротко заверши мысль в уже отправленном или промолчи."
                        .to_string(),
                    Vec::new(),
                );
            }
            // Живая очередь сообщений: цель — 1-2 части; жёсткий потолок —
            // cfg.misc.max_message_parts (3). Лучше меньше, чем поток пингов.
            // Шаг 1 — склейка мелких частей: если оба соседних фрагмента короткие
            // (<40 симв) — сливаем в одно сообщение (до ~160 симв), чтобы вместо
            // 4 пшиков приходила 1 нормальная реплика.
            let raw_parts = parts;
            let mut merged: Vec<String> = Vec::new();
            for p in raw_parts {
                let p = p.to_string();
                match merged.last_mut() {
                    Some(last) if last.chars().count() < 40 && p.chars().count() < 40
                        && last.chars().count() + p.chars().count() <= 160 =>
                    {
                        last.push('\n');
                        last.push_str(&p);
                    }
                    _ => merged.push(p),
                }
            }
            let parts: Vec<_> = merged
                .into_iter()
                .take(cfg.misc.max_message_parts.max(1).min(parts_remaining))
                .collect();
            // Анти-клише: дежурные формулы («я тут, рядом», «хочешь, чтобы
            // я…», «расскажи ещё», «застряла в мыслях») — это не ответ на его
            // слова. Выкидываем ТОЛЬКО такие части, живые оставляем. Если
            // клише оказались все — возвращаем ошибку с указанием на его слова.
            let mut kept: Vec<String> = Vec::new();
            let mut dropped: Vec<(&'static str, String)> = Vec::new();
            for part in parts {
                if let Some(why) = cliche_hit(&part).or_else(|| fragment_hit(&part)) {
                    warn!("[tg] анти-клише/обрубок выкинул часть: {why} — {:?}", part);
                    rep.remember_blocked(&part);
                    dropped.push((why, part));
                } else {
                    kept.push(part);
                }
            }
            if kept.is_empty() {
                let why = dropped.first().map(|(w, _)| *w).unwrap_or("дежурная формула");
                let quote: String = his_last.trim().chars().take(180).collect::<String>();
                return (
                    format!(
                        "НЕ ОТПРАВЛЕНО: {why}. Так ты не отвечаешь на его слова, а подставляешь \
                         готовую формулу или обрубок. Он только что написал: «{quote}». Ответь на \
                         КОНКРЕТНОЕ в этих словах — что он сказал или спросил, что это \
                         значит для тебя, что ты об этом думаешь. Своими словами, без \
                         утешений, без «я рядом/я тут», без вопросов «хочешь, чтобы я…» \
                         и без «расскажи ещё». Если не знаешь, что сказать, — скажи \
                         прямо, что не знаешь, а не подставляй шаблон."
                    ),
                    Vec::new(),
                );
            }
            let parts = kept;
            // Анти-повтор (порт kuni): черновик слишком похож на её недавние
            // реплики — не отправляем ВООБЩЕ (иначе повторится спарки «Мурр…
            // я тут» из хода в ход). Модель увидит anti_repeat-подсказку и
            // переформулирует или промолчит. Сравниваем и с частями ЭТОЙ пачки
            // (part2 против part1), иначе модель шлёт два парафраза подряд.
            // И — с черновиками, уже заблокированными в этом ходу: они не ушли
            // в чат (нет в истории), но модель могла отправить их же снова.
            let mut seen: Vec<String> = own_recent.to_vec();
            seen.extend(rep.blocked.iter().cloned());
            let mut batch: Vec<String> = Vec::new();
            for part in &parts {
                // Внутри одной пачки — строже: part2 не должен быть парафразом
                // part1 (две «разные» реплики несут одну мысль).
                if let Some(prev) = batch
                    .iter()
                    .find(|p| lexical_similarity(part, p) > PARTS_REPEAT_THRESHOLD)
                {
                    let sim = lexical_similarity(part, prev);
                    warn!(
                        "[tg] анти-повтор (части ответа): {sim:.2} — {:?} ≈ {:?}",
                        part, prev
                    );
                    rep.remember_blocked(part);
                    return (
                        format!(
                            "ПОВТОР ({sim:.2}): это та же мысль, что в другой части ответа. \
                             Оставь ОДНУ часть с этой мыслью, а вторую сделай про другое — \
                             или не отправляй вторую вовсе."
                        ),
                        Vec::new(),
                    );
                }
                if let Err(sim) = anti_repeat_check(part, &seen, diary, cfg, rep).await {
                    warn!("[tg] анти-повтор заблокировал отправку: {sim} — {:?}", part);
                    rep.remember_blocked(part);
                    return (
                        format!(
                            "ПОВТОР ({sim}). Тот же смысл ты уже отправляла. {anti_repeat_prompt}"
                        ),
                        Vec::new(),
                    );
                }
                batch.push(part.clone());
                seen.push(part.clone());
            }
            let mut rng = rand::thread_rng();
            for part in &parts {
                // Реалистичный ввод текста: typing-action + пауза по длине
                // (было: фиксированные 400ms → выглядели механически-быстро).
                let chars = part.chars().count() as u64;
                let typing_ms = (600 + 45 * chars).clamp(800, 3200)
                    .saturating_add(rng.gen_range(0..500));
                client.send_action_typing(chat_id);
                tokio::time::sleep(Duration::from_millis(typing_ms)).await;
                // Живость: с вероятностью misc.liveliness_typo_probability
                // отправляем с одной опечаткой и тут же редактируем на верный
                // текст («перепечатала»).
                // В контекст запись идёт КОРРЕКТНЫМ текстом (parts не тащим).
                let typo_p = cfg.misc.liveliness_typo_probability;
                let to_send = if typo_p > 0.0 && rng.gen_bool(typo_p.min(1.0) as f64) {
                    match crate::corrupt_text_once(part) {
                        Some(bad) if bad != *part => {
                            client.queue_typo_fix(chat_id, part.clone());
                            info!("[tg] опечатка: {:?}", bad);
                            bad
                        }
                        _ => part.clone(),
                    }
                } else {
                    part.clone()
                };
                client.send_text(chat_id, &to_send);
                info!("[tg] → отправлено в чат: {:?}", to_send);
                // Отбивка между сообщениями: человек не шлёт пачку мгновенно.
                tokio::time::sleep(Duration::from_millis(rng.gen_range(250..800))).await;
            }
            (
                format!("Отправлено {} сообщений в Telegram.", parts.len()),
                parts,
            )
        }
        "send_gif" => {
            let v: Value = match serde_json::from_str(args) {
                Ok(v) => v,
                Err(e) => return (format!("Ошибка парсинга send_gif: {e}"), Vec::new()),
            };
            let query = v
                .get("query")
                .and_then(|q| q.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if query.is_empty() {
                return ("ОШИБКА: send_gif получил пустой query.".to_string(), Vec::new());
            }
            info!("[tg] поиск GIF: {:?}", query);
            client.send_action_typing(chat_id);
            match client.search_and_send_gif(chat_id, &query).await {
                Ok(true) => (
                    "GIF отправлена в чат.".to_string(),
                    vec![format!("[я отправила GIF (поиск: {query})]")],
                ),
                Ok(false) => (
                    "GIF по запросу не найдена — ответь просто текстом.".to_string(),
                    Vec::new(),
                ),
                Err(e) => {
                    error!("[tg] send_gif ошибка: {}", e);
                    (
                        format!("GIF не отправилась ({e}) — ответь просто текстом."),
                        Vec::new(),
                    )
                }
            }
        }
        "react_with_emoji" => {
            let v: Value = match serde_json::from_str(args) {
                Ok(v) => v,
                Err(e) => return (format!("Ошибка парсинга react_with_emoji: {e}"), Vec::new()),
            };
            let emoji = v
                .get("emoji")
                .and_then(|e| e.as_str())
                .unwrap_or("👍")
                .to_string();
            client.add_reaction(chat_id, message_id, &emoji);
            (format!("Реакция {emoji} поставлена."), Vec::new())
        }
        "ask" => {
            let query_text = extract_query_arg(args).unwrap_or_default();
            if query_text.is_empty() {
                return ("Дневник: пустой запрос #ask.".to_string(), Vec::new());
            }
            let res: String = match diary.embedding().embed(&query_text).await {
                Ok(vec) => {
                    let hits = diary.query_top_n(diary_entries, &vec, 6);
                    if hits.is_empty() {
                        "Дневник: релевантных воспоминаний не найдено.".to_string()
                    } else {
                        let mut out = String::from("Дневник (топ релевантных):\n");
                        for h in &hits {
                            let preview: String = h
                                .entry
                                .body
                                .chars()
                                .take(1200)
                                .collect();
                            let preview = preview.trim();
                            out.push_str(&format!(
                                "[{:.2}] {}…\n{}\n---\n",
                                h.relatedness,
                                preview.lines().next().unwrap_or(preview),
                                preview
                            ));
                        }
                        out
                    }
                }
                Err(e) => format!("Дневник: ошибка эмбеддинга запроса: {e}"),
            };
            (res, Vec::new())
        }
        "situation" => {
            // Модель сама достаёт фрагмент ситуационной памяти: конкретную
            // заметку (number), последние подряд (last) и/или похожее по
            // смыслу (query). Свежие заметки и так уже в контексте.
            let v: Value = serde_json::from_str(args).unwrap_or(Value::Null);
            let number = v.get("number").and_then(|x| x.as_u64());
            let last = v.get("last").and_then(|x| x.as_u64());
            let query = v
                .get("query")
                .and_then(|q| q.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty());

            let mut chunks: Vec<String> = Vec::new();
            if let Some(n) = number {
                match situation.block_by_number(n as u32) {
                    Some(b) => chunks.push(b),
                    None => chunks.push(format!("Заметки №{n} нет.")),
                }
            }
            if let Some(k) = last {
                let r = situation.recent(k as usize);
                if !r.trim().is_empty() {
                    chunks.push(r);
                }
            }
            if let Some(q) = query {
                let r = situation
                    .search(diary.embedding(), q, cfg.misc.situation_top_k.max(3))
                    .await;
                if !r.trim().is_empty() {
                    chunks.push(format!("[похожее по смыслу на «{q}»]\n{r}"));
                }
            }
            if chunks.is_empty() {
                let r = situation.recent(5);
                chunks.push(if r.trim().is_empty() {
                    "Ситуационных заметок пока нет.".to_string()
                } else {
                    r
                });
            }
            (
                format!("Ситуационная память:\n{}", chunks.join("\n\n")),
                Vec::new(),
            )
        }
        "stay_quiet" => {
            // Модель САМА решила промолчать. В чат не уходит ничего —
            // это легальный исход idle-хода (таймер дал шанс, а она им
            // не воспользовалась).
            info!("[tg] модель промолчала (stay_quiet)");
            (
                "ОК: ты промолчала. В чат ничего не отправлено, ход завершён.".to_string(),
                Vec::new(),
            )
        }
        "internet_search" => {
            let v: Value = match serde_json::from_str(args) {
                Ok(v) => v,
                Err(e) => {
                    return (
                        format!("ОШИБКА: невалидный JSON в internet_search: {e}"),
                        Vec::new(),
                    )
                }
            };
            let query = v["query"].as_str().unwrap_or("").trim();
            if query.is_empty() {
                return (
                    "ОШИБКА: пустой запрос internet_search.".to_string(),
                    Vec::new(),
                );
            }
            // Два источника параллельно: веб (DDG) и новости по теме (Google News).
            // DDG душит челленджами resident IP — тогда остаются новости.
            let (web_res, news_res) = tokio::join!(
                furinya_web::search(query, 4),
                furinya_web::news_search(query, 4),
            );
            let mut out = format!("Результаты поиска «{query}»:\n");
            let mut found = false;
            if let Ok(list) = web_res.as_ref() {
                for r in list.iter() {
                    found = true;
                    let mut sn: String = r.snippet.chars().take(180).collect();
                    if r.snippet.chars().count() > 180 {
                        sn.push('…');
                    }
                    out.push_str(&format!("· {} — {}\n   {}\n", r.title, r.url, sn));
                }
            }
            if let Ok(list) = news_res.as_ref() {
                for n in list.iter() {
                    found = true;
                    let date = if n.date_str.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", n.date_str)
                    };
                    out.push_str(&format!("📰 {} — {}{}\n", n.title, n.source, date));
                }
            }
            if found {
                (out, Vec::new())
            } else {
                let err = web_res
                    .err()
                    .map(|e| e.to_string())
                    .or_else(|| news_res.err().map(|e| e.to_string()))
                    .unwrap_or_else(|| "пустые результаты".to_string());
                (
                    format!(
                        "ОШИБКА поиска: {err}. Можешь честно сказать собеседнику, что у тебя интернет глючит, — желательно с грустью в голосе."
                    ),
                    Vec::new(),
                )
            }
        }
        "web_open" => {
            let v: Value = match serde_json::from_str(args) {
                Ok(v) => v,
                Err(e) => {
                    return (
                        format!("ОШИБКА: невалидный JSON в web_open: {e}"),
                        Vec::new(),
                    )
                }
            };
            let url = v["url"].as_str().unwrap_or("").trim();
            if url.is_empty() || !(url.starts_with("http://") || url.starts_with("https://")) {
                return (
                    "ОШИБКА: web_open ждёт ссылку, начинающуюся с http:// или https://."
                        .to_string(),
                    Vec::new(),
                );
            }
            match furinya_web::open_page(url, 2000).await {
                Ok(text) => (text, Vec::new()),
                Err(e) => (
                    format!("ОШИБКА открытия {url}: {e}"),
                    Vec::new(),
                ),
            }
        }
        "fresh_news" => {
            let headlines = news_headlines(12);
            if headlines.is_empty() {
                return (
                    "Новостной кэш пока пуст (каналы ещё не опросились) — попробуй потом или используй internet_search."
                        .to_string(),
                    Vec::new(),
                );
            }
            let mut out = String::from("Свежее из новостных каналов:\n");
            for it in &headlines {
                out.push_str(&format_news_line(it, 160));
                out.push('\n');
            }
            (out, Vec::new())
        }
        "diary_page" => (
            "Инструмент diary_page недоступен в чате. Память консолидируется автоматически в режиме сна."
                .to_string(),
            Vec::new(),
        ),
        "generate_image" => {
            if !cfg.capabilities.take_photo.enabled {
                return (
                    "Инструмент generate_image отключён в конфиге (take_photo.enabled=false)."
                        .to_string(),
                    Vec::new(),
                );
            }
            let v: Value = match serde_json::from_str(args) {
                Ok(v) => v,
                Err(e) => {
                    return (
                        format!("ОШИБКА: невалидный JSON в generate_image: {e}"),
                        Vec::new(),
                    )
                }
            };
            let raw_prompt = v
                .get("prompt")
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if raw_prompt.is_empty() {
                return (
                    "ОШИБКА: generate_image получил пустой prompt.".to_string(),
                    Vec::new(),
                );
            }
            let explicit_negative = v
                .get("negative_prompt")
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            // NSFW — только явным флагом от модели (приватный чат с Сашей).
            // Если модель забыла указать, но промпт открытый — верим промпту.
            let mut nsfw = v.get("nsfw").and_then(|p| p.as_bool()).unwrap_or(false);
            let p_lower = raw_prompt.to_lowercase();
            if !nsfw
                && (p_lower.contains("nsfw")
                    || p_lower.contains("nude")
                    || p_lower.contains("naked")
                    || p_lower.contains("голая")
                    || p_lower.contains("обнажен")
                    || p_lower.contains("секс")
                    || p_lower.contains("эроти"))
            {
                nsfw = true;
            }
            // Канон: тело + внешность. Глаза принудительно усиливаем до
            // гетерохромии 1.5 — иначе на широких/боковых кадрах она бледная.
            // Канон идёт в промпт ВСЕГДА и не зависит от LLM.
            let body_tags = std::fs::read_to_string("prompts/character_body.md")
                .unwrap_or_default()
                .trim()
                .trim_end_matches(['.', ','])
                .to_string();
            let canon_tags = std::fs::read_to_string("prompts/character_appearance.md")
                .unwrap_or_default()
                .trim()
                .trim_end_matches(['.', ','])
                .to_string();
            let canon = [body_tags, strengthen_eyes(&canon_tags)]
                .into_iter()
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(", ");
            // NSFW-термины вырезаем перед инженером (LLM может отказаться),
            // вернём их в позитивный промпт после (трюк kuni).
            let scene_for_llm = if nsfw {
                sanitize_for_llm(&raw_prompt)
            } else {
                raw_prompt.clone()
            };
            info!(
                "[tg] generate_image: сцена от LLM={:?} (nsfw={nsfw})",
                raw_prompt
            );
            // LLM добавляет ТОЛЬКО сцену: действие, ракурс, одежду, свет.
            let (scene_positive, engineered_negative) =
                match engineer_sd_prompt(llm, &scene_for_llm, nsfw).await {
                    Ok((p, n)) => (p, n),
                    Err(e) => {
                        warn!(
                            "[tg] generate_image: prompt-инженер упал ({e}) — fallback на прямой конкат"
                        );
                        (scene_for_llm.clone(), String::new())
                    }
                };
            let mut positive = dedupe_exact_tags(&format!("{canon}, {scene_positive}"));
            // NSFW-термины из исходного описания возвращаем в промпт (их не
            // видел prompt-инженер, иначе мог отказаться).
            if nsfw {
                positive = append_explicit_words(&positive, &raw_prompt);
            }
            // Негатив: явный аргумент модели → негатив инженера → дефолт.
            let negative_prompt = if !explicit_negative.is_empty() {
                explicit_negative
            } else if !engineered_negative.trim().is_empty() {
                engineered_negative
            } else {
                DEFAULT_SD_NEGATIVE.to_string()
            };
            // Куда сохраняем сгенерированный PNG.
            let out_dir = std::env::temp_dir();
            let fname = format!(
                "furinya_gen_{}.png",
                Utc::now().format("%Y%m%d_%H%M%S")
            );
            let out_path = out_dir.join(&fname);
            info!(
                "[tg] generate_image: positive={:?} negative={:?} → бэкенд {} (nsfw={nsfw})",
                positive, negative_prompt, cfg.capabilities.take_photo.backend
            );
            client.send_action_typing(chat_id);
            match generate_image(
                &cfg.capabilities.take_photo,
                &positive,
                &negative_prompt,
                nsfw,
                &out_path,
            )
            .await
            {
                Ok(path) => {
                    // Отправляем картинку в чат.
                    client.send_photo(chat_id, &path);
                    info!("[tg] generate_image: отправлено фото {}", path);
                    (
                        format!("Изображение сгенерировано и отправлено: {path}"),
                        vec![format!("[я отправила ФОТО (промпт): {positive}]")],
                    )
                }
                Err(e) => {
                    error!("[tg] generate_image ошибка: {}", e);
                    (
                        format!("ОШИБКА генерации изображения: {e}"),
                        Vec::new(),
                    )
                }
            }
        }
        "send_voice_message" => {
            let rv = &cfg.capabilities.record_voice;
            if !rv.enabled {
                return (
                    "Инструмент send_voice_message отключён в конфиге (record_voice.enabled=false) — используй send_telegram_message."
                        .to_string(),
                    Vec::new(),
                );
            }
            let v: Value = match serde_json::from_str(args) {
                Ok(v) => v,
                Err(e) => {
                    return (
                        format!("ОШИБКА: невалидный JSON в send_voice_message: {e}"),
                        Vec::new(),
                    )
                }
            };
            // Текст для озвучки: вырезаем служебные строки и эмодзи — TTS их
            // прочитает вслух мусором.
            let raw_text = v
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            let mut voice_text = strip_meta_lines(&raw_text);
            // Ролевые действия в *звёздочках* → в озвучку не годятся.
            voice_text = strip_asterisk_actions(&voice_text);
            // Вырезаем эмодзи/декор-символы: TTS прочитает их мусором или зависнет.
            voice_text.retain(|c| {
                let cp = c as u32;
                !( (0x1F000..=0x1FAFF).contains(&cp)      //Emoji: основной блок и дополнения
                    || (0x2600..=0x27BF).contains(&cp)     //Misc symbols / dingbats
                    || (0x2190..=0x21FF).contains(&cp)     //Стрелки
                    || (0x1F300..=0x1F5FF).contains(&cp)   //Misc symbols and pictographs
                    || (0x1F600..=0x1F64F).contains(&cp)   //Emoticons
                    || cp == 0xFE0F                        //Variation selector-16
                    || cp == 0x200D )                      //Zero-width joiner
            });
            let mut voice_text = voice_text.trim().to_string();
            if voice_text.is_empty() {
                return (
                    "ОШИБКА: send_voice_message получил пустой text.".to_string(),
                    Vec::new(),
                );
            }
            // Анти-повтор: то же самое для голосовых (ночные обнимашки шли
            // одной и той же фразой). Учитываем и ранее заблокированные
            // черновики этого хода — иначе модель повторяла заблокированную
            // фразу следующим вызовом.
            let mut voice_seen: Vec<String> = own_recent.to_vec();
            voice_seen.extend(rep.blocked.iter().cloned());
            if let Err(sim) = anti_repeat_check(&voice_text, &voice_seen, diary, cfg, rep).await {
                warn!("[tg] анти-повтор заблокировал голосовое: {sim}");
                rep.remember_blocked(&voice_text);
                return (
                    format!(
                        "ПОВТОР ({sim}). То же самое ты уже отправляла голосом. {anti_repeat_prompt}"
                    ),
                    Vec::new(),
                );
            }
            if voice_text.len() > 900 {
                // ~1.5 лимита голосового: берём начало.
                voice_text = voice_text.chars().take(900).collect();
            }
            let out_dir = std::env::temp_dir();
            let fname = format!("furinya_voice_{}.mp3", Utc::now().format("%Y%m%d_%H%M%S%3f"));
            let out_path = out_dir.join(&fname);
            client.send_action_typing(chat_id);

            let backend = rv.backend.trim().to_lowercase();
            let synth_res = match backend.as_str() {
                "fish" if !rv.fish.reference_id.trim().is_empty() => {
                    synthesize_fish_voice(&rv.fish, &voice_text, &out_path).await
                }
                "openai" if !rv.openai.url.trim().is_empty() => {
                    synthesize_speech_openai(&rv.openai, &voice_text, &out_path).await
                }
                "fish" => Err(anyhow::anyhow!(
                    "backend=fish, но пусто fish.referenceId — создай голосовую модель Furinya \
                     (docs/voice-reference-guide.md) или переключись на backend=\"openai\" \
                     (локальный бесплатный XTTS)"
                )),
                "openai" => Err(anyhow::anyhow!(
                    "backend=openai, но пусто record_voice.openai.url — подними локальный \
                     XTTS-сервер (openedai-speech) и пропиши его URL в config.toml"
                )),
                other => Err(anyhow::anyhow!(
                    "неизвестный TTS-бэкенд '{other}' (доступны: fish, openai)"
                )),
            };
            match synth_res {
                Ok(path) => {
                    client.send_voice_note(chat_id, &path, 0);
                    info!("[tg] send_voice_message: отправлено голосовое {}", path);
                    (
                        "Голосовое сообщение отправлено.".to_string(),
                        vec![format!("[я отправила ГОЛОСОВОЕ: {voice_text}]")],
                    )
                }
                Err(e) => {
                    error!("[tg] send_voice_message ошибка: {}", e);
                    (
                        format!(
                            "ОШИБКА синтеза речи: {e}. Отправь ответ текстом через send_telegram_message."
                        ),
                        Vec::new(),
                    )
                }
            }
        }
        other => (
            format!("Инструмент '{other}' не реализован в TG-режиме."),
            Vec::new(),
        ),
    }
}

/// Обрабатывает одно входящее сообщение (или idle-стимул) агентным циклом.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::collapsible_if)]
#[allow(clippy::ptr_arg)]
pub async fn handle_new_message(
    v: Value,
    llm: &LlmClient,
    client: &TdClient,
    diary: &Diary,
    prompts: &Prompts,
    diary_entries: &mut Vec<Entry>,
    tg_history: &mut Vec<Message>,
    tg_history_full: &mut Vec<(i64, bool, String, i64)>,
    tg_system_prompt: &str,
    cfg: &Config,
    my_id: i64,
    is_idle: bool,
) {
    let msg = &v["message"];
    let chat_id = msg["chat_id"].as_i64().unwrap_or(0);
    let message_id = msg["id"].as_i64().unwrap_or(0);
    let content_type = msg["content"]["@type"].as_str().unwrap_or("?");
    info!("[tg][debug] handle_new_message: content_type={} chat_id={} sender={}", content_type, chat_id, msg["sender_id"]["user_id"].as_i64().unwrap_or(0));
    let text = msg["content"]["text"]["text"]
        .as_str()
        .or_else(|| msg["content"]["caption"]["text"].as_str())
        .unwrap_or("")
        .to_string();

    let sender = msg["sender_id"]["user_id"].as_i64().unwrap_or(0);
    let sender_name = if sender == my_id {
        cfg.general.character_name.clone()
    } else if sender == cfg.general.papik_chat_id {
        cfg.general.papik_name.clone()
    } else {
        format!("user{}", sender)
    };

    // Свои исходящие сообщения НЕ обрабатываем: иначе бот отвечает сам себе,
    // скачивает и «распознаёт» СОБСТВЕННЫЕ голосовые (засоряя Whisper и
    // контекст) и путается в диалоге. ВАЖНО: этот guard раньше стоял ПОСЛЕ
    // веток фото/голоса — и бот успевал обработать собственный голос.
    if !is_idle && sender == my_id {
        debug!("[tg] пропускаю собственное исходящее сообщение (sender={})", sender);
        return;
    }

    // ── Ситуационная память этого чата ────────────────────────────────────
    // Атомарная последовательная история «что происходит между нами прямо
    // сейчас» (data/diary/situation/<chat_id>.md). Читаем на turn 1, пишем
    // одну заметку после обмена (в конце функции).
    let situation = SituationStore::new(&cfg.misc.diary_dir, chat_id);
    // Длина полной истории на входе — по ней в конце поймём, был ли реальный
    // обмен (стоит ли писать заметку). Считаем ДО добавления его сообщения.
    let hist_len_before = tg_history_full.len();

    // Живой статус: приходит живое сообщение от человека → на минуту "В сети".
    // Через тишину статус снова сбрасывается (см. timeout-ветку главного цикла).
    // Для idle-хода статус НЕ поднимаем: если она промолчит — она и не «в сети»
    // (иначе мигание «в сети / не в сети» на пустом месте). Если написала —
    // статус поднимет вызывающий код, по факту отправки.
    if !is_idle {
        client.set_online(true);
    }

    // ── Ветка ФОТО (распознавание от юзера) ───────────────────────────────
    // messagePhoto: качаем photo.sizes[-1] (самый большой доступный размер,
    // TDLib сам даёт разумный — не хардкодим), описываем через Gemma multimodal.
    // Если к фото есть подпись (text) — оставляем её; иначе text = описание.
    let mut photo_description: Option<String> = None;
    if cfg.capabilities.vision.enabled
        && msg["content"]["@type"].as_str() == Some("messagePhoto")
    {
        if let Some(sizes) = msg["content"]["photo"]["sizes"].as_array() {
            if let Some(big) = sizes.last() {
                let file_id = big["photo"]["id"].as_i64().unwrap_or(0);
                if file_id != 0 {
                    info!("[tg] фото от {}: скачиваю file_id={}", sender_name, file_id);
                    client.send_action_typing(chat_id);
                    match client.download_file(file_id, 16).await {
                        Ok(path) => {
                            let vcfg = &cfg.capabilities.vision;
                            let vllm = LlmClient::new(
                                &vcfg.llm_image_to_text.base_url,
                                &vcfg.llm_image_to_text.bearer_key,
                                &vcfg.llm_image_to_text.model,
                                furinya_llm::SamplingParams {
                                    // Vision-описания — служебные, холодные
                                    // (факт по картинке, а не импровизация).
                                    temperature: 0.3,
                                    top_p: None,
                                    top_k: None,
                                    min_p: cfg.misc.llm_min_p.parse().ok(),
                                    presence_penalty: cfg.misc.presence_penalty.parse().ok(),
                                    repetition_penalty: None,
                                },
                            );
                            match describe_photo(
                                &vllm,
                                vcfg,
                                &path,
                                "Опиши это фото кратко и по делу (по-русски). Люди, объекты, \
                                 контекст, настроение. Не выдумывай того, чего не видно.",
                            )
                            .await
                            {
                                Ok(desc) if !desc.trim().is_empty() => {
                                    info!("[tg] фото описано ({} симв.): {}", desc.len(), desc);
                                    photo_description = Some(desc);
                                }
                                Ok(_) => warn!("[tg] vision вернул пустое описание"),
                                Err(e) => error!("[tg] vision ошибка: {}", e),
                            }
                        }
                        Err(e) => error!("[tg] не удалось скачать фото: {}", e),
                    }
                }
            }
        }
    }

    // ── Ветка ГОЛОС (расшифровка от юзера) ──────────────────────────────────
    // messageVoiceNote: качаем voice_note.voice.file.id, распознаём через
    // локальный Whisper (CPU — медленно, поэтому typing-экшен всё время).
    // Whisper не идеален — добавляем сноску о неуверенности, чтобы модель
    // не "зацикливалась" на точности и не переспрашивала лишний раз.
    let mut voice_text: Option<String> = None;
    let ctype = msg["content"]["@type"].as_str().unwrap_or("?");
    if ctype != "messageText" && ctype != "messagePhoto" {
        info!("[tg][debug] неизвестный/нетекстовый контент: type={} content={}", ctype, msg["content"]);
    }
    if cfg.capabilities.hearing.enabled
        && msg["content"]["@type"].as_str() == Some("messageVoiceNote")
    {
        info!("[tg][debug] голос: нашёл messageVoiceNote, content={}", msg["content"]);
        let file_id = msg["content"]["voice_note"]["voice"]["id"]
            .as_i64()
            .unwrap_or(0);
        let duration = msg["content"]["voice_note"]["duration"].as_i64().unwrap_or(0);
        if file_id != 0 {
            info!(
                "[tg] голос от {}: скачиваю file_id={} (длит {}с)",
                sender_name, file_id, duration
            );
            client.send_action_typing(chat_id);
            match client.download_file(file_id, 16).await {
                Ok(path) => {
                    // Whisper на CPU может крутить долго — постоянно держим typing.
                    client.send_action_typing(chat_id);
                    match transcribe_audio(&cfg.capabilities.hearing.llm_audio_to_text, std::path::Path::new(&path), Some("ru")).await {
                        Ok(t) if !t.trim().is_empty() => {
                            info!("[tg] голос распознан ({} симв.): {}", t.len(), t);
                            voice_text = Some(t);
                        }
                        Ok(_) => warn!("[tg] Whisper вернул пустую расшифровку"),
                        Err(e) => error!("[tg] Whisper ошибка: {}", e),
                    }
                }
                Err(e) => error!("[tg] не удалось скачать голос: {}", e),
            }
        }
    }

    let date = msg["date"].as_i64().unwrap_or(0);
    info!("[tg] NEW MSG chat={} sender={}", chat_id, sender);
    // «Эффективный текст» сообщения: подпись текста > описание фото >
    // расшифровка голоса. Вычисляем ОДИН раз до агентного цикла.
    let effective_text = if photo_description.is_some() {
        // ФОТО (+ подпись): описание обязательно попадает в контекст.
        // Раньше при подписи описание ВЫБАДЫВАЛОСЬ — бот «не видел» картинку.
        let ph = photo_description.clone().unwrap_or_default();
        if !text.trim().is_empty() {
            format!(
                "[Пользователь прислал ФОТО с подписью: «{}». Описание того, что на нём: {}]",
                text.trim(),
                ph
            )
        } else {
            format!("[Пользователь прислал ФОТО. Описание того, что на нём: {}]", ph)
        }
    } else if !text.trim().is_empty() {
        text.clone()
    } else if let Some(vt) = &voice_text {
        // Сноска о неуверенности Whisper (CPU, может ошибаться) и ЖЁСТКОЕ
        // правило ответа: на голосовое отвечаем ТОЛЬКО голосом (guard в
        // tool-цикле блокирует send_telegram_message при voice_input).
        format!(
            "[Пользователь прислал ГОЛОСОВОЕ сообщение. Расшифровка (Whisper, \
             возможны неточности — не переспрашивай ради проверки, лучше \
             уточни по смыслу, если совсем непонятно): {}]. \
             ВАЖНО: на голосовое отвечай ТОЛЬКО голосом — инструментом \
             send_voice_message (одна живая реплика, как будто говоришь в чат). \
             НИКАКОГО текста в чат — send_telegram_message ЗАПРЕЩЁН.",
            vt
        )
    } else {
        String::new()
    };
    if !is_idle && !effective_text.is_empty() {
        // Живая память: сообщение юзера попадает в контекст чата СРАЗУ
        // (раньше текущий разговор не сохранялся между сообщениями —
        // бот «не помнил» ничего из того, что происходило после старта).
        let ts0 = format_ts(date);
        remember_in_history(
            tg_history,
            Message::user(format!("[{}] {}: {}", ts0, sender_name, effective_text)),
        );
        tg_history_full.push((sender, false, effective_text.clone(), date));
    }
    let ts = format_ts(date);
    println!("\n[tg] {} [{}]: {}", sender_name, ts, effective_text);

    // Его слова БЕЗ служебных обёрток (фото/голос) — для оценки хода.
    let his_words = if !text.trim().is_empty() {
        text.clone()
    } else {
        photo_description
            .clone()
            .or_else(|| voice_text.clone())
            .unwrap_or_default()
    };

    // ── Оценка хода (внутренний разбор перед ответом) ─────────────────────
    // kuni получал оценку состояния/намерения собеседника прямо в ответе
    // (`- thought:`/`- emotion:`). Rust-порт ждал native `reasoning_content`,
    // которого `ministral-8b-2512` НЕ отдаёт → этап оценки просто исчез, и
    // бот отвечал generic-заглушками («я тут немного зависла…»).
    //
    // Поэтому оценка — ОТДЕЛЬНЫЙ дешёвый LLM-вызов (temp 0.3, без tools):
    // модель возвращает три строки — СОСТОЯНИЕ ОН / СОСТОЯНИЕ Я / ПЛАН.
    // Дальше блок инжектится в system-промпт хода («Оценка хода»), а per-chat
    // состояние пишется в data/diary/chat_state/<chat_id>.md.
    let assessment: Option<Assessment> = if !is_idle && !his_words.trim().is_empty() {
        let assess_prompt = prompts.assess();
        if assess_prompt.trim().is_empty() {
            warn!("[tg] промпт оценки (assess.md) отсутствует — ход без оценки");
            None
        } else {
            let diary_dir = std::path::Path::new(&cfg.misc.diary_dir);
            let prev_state = read_chat_state(diary_dir, chat_id);
            // Хвост истории: последние реплики (её и его) ДО его текущего
            // сообщения — оно идёт в оценку отдельно, без обёрток (голос/фото).
            let hist_end = tg_history_full.len().saturating_sub(1);
            let hist_start = hist_end.saturating_sub(6);
            let hist_tail = situation_exchange_text(cfg, &tg_history_full[hist_start..hist_end]);
            let input = build_assess_input(&his_words, &hist_tail, &prev_state);
            // Пока идёт оценка — показываем «печатает…»: иначе пользователь
            // видит тишину несколько секунд и думает, что бот завис.
            client.send_action_typing(chat_id);
            let assess_llm = llm.with_temperature(0.3);
            let call = assess_llm
                .chat(vec![Message::system(assess_prompt), Message::user(input)], None);
            match tokio::time::timeout(Duration::from_secs(20), call).await {
                Ok(Ok(resp)) => {
                    let raw = resp
                        .choices
                        .first()
                        .and_then(|c| match &c.message.content {
                            MessageContent::Text(s) => Some(s.clone()),
                            MessageContent::Parts(_) => None,
                        })
                        .unwrap_or_default();
                    match parse_assessment(&raw) {
                        Some(a) => {
                            info!(
                                "[tg] ОЦЕНКА ХОДА:\n  ОН: {}\n  Я: {}\n  ПЛАН: {}",
                                a.he, a.me, a.plan
                            );
                            Some(a)
                        }
                        None => {
                            warn!("[tg] оценка хода: модель не вернула 3 строки — ход без оценки");
                            None
                        }
                    }
                }
                Ok(Err(e)) => {
                    warn!("[tg] оценка хода: LLM ошибка ({e}) — ход без оценки");
                    None
                }
                Err(_) => {
                    warn!("[tg] оценка хода: таймаут 20с — ход без оценки");
                    None
                }
            }
        }
    } else {
        None
    };
    // System-промпт хода: базовый + внутренняя оценка (её видит и на
    // последующих итерациях tool-loop).
    let system_prompt = match &assessment {
        Some(a) => format!("{}\n\n{}", tg_system_prompt, format_assessment(a)),
        None => tg_system_prompt.to_string(),
    };

    // Технические команды (consolidate/save/exit) НЕ обрабатываются в TG —
    // только в локальный stdin (см. main.rs). В TG игнорируем их полностью.
    //
    // Автоматический 👍 УБРАН: реакция «на каждое сообщение» была механической
    // и обесценивалась. Модель сама решает через react_with_emoji (см.
    // описание инструмента) — ставить ли реакцию и какую по настроению.

    let max_turns = 8;
    let mut ask_exhausted = false;
    let mut final_sent_via_tool = false;
    // Сколько ЧАСТЕЙ (реальных сообщений) ушло за ход — жёсткий гейт против
    // «потопа» (раньше после анти-повтора ход шёл до max_turns и слал 2+2+3).
    let mut parts_sent = 0usize;
    // Модель САМА выбрала промолчать (stay_quiet) — ход закрывается тишиной.
    let mut chose_quiet = false;
    // Инструмент не выполнен: stay_quiet на живом сообщении, текст на голосовое,
    // кривой JSON в send_telegram_message. Ход в этом случае НЕ закрываем —
    // даём модели ещё один ход ответить. Раньше здесь падало в canned-заглушку
    // («Сашенька, я чуть зависла…») — ровно тот дежурный ответ, который чиним.
    let mut tool_blocked = false;
    // Инструментальный диалог ЭТОГО вызова (assistant tool_calls → tool-результаты).
    // Ничего с role=tool не должно попадать в tg_history: «осиротевший» tool
    // без предшествующего assistant.tool_calls роняет Mistral ошибкой
    // 400 "Unexpected role 'tool' after role 'system'" (код 3230).
    let mut turn_ctx: Vec<Message> = Vec::new();
    // Анти-повтор (порт kuni): кэш эмбеддингов живёт на один вызов хода —
    // черновики отправок сравниваются с её недавними репликами из истории.
    let mut rep_state = AntiRepeatState::new();
    let anti_repeat_prompt = prompts
        .extra
        .get("anti_repeat")
        .cloned()
        .unwrap_or_default();
    for turn in 1..=max_turns {
        // Ход, в котором анти-повтор вернул ПОВТОР-нудж: canned-фолбэки после
        // него запрещены (тишина/переписывание лучше «я чуть зависла…» — это
        // тоже спам). Нудж живёт ОДИН ход — объявляем его здесь, на каждой
        // итерации, иначе он тащится дальше и гонит цикл до max_turns
        // (из-за этого бот слал пачки 2+2+3 сообщений подряд).
        let mut repeat_nudged = false;
        let mut messages = vec![Message::system(system_prompt.clone())];
        if turn == 1 {
            // RAG-запрос: текст, а если его нет — описание фото/расшифровка голоса.
            let rag_query = if !text.trim().is_empty() {
                text.clone()
            } else {
                photo_description
                    .clone()
                    .or_else(|| voice_text.clone())
                    .unwrap_or_default()
            };
            // ── Бюджет памяти в контексте ─────────────────────────────
            // diary_injection_max_length — сколько символов памяти влезает
            // в один запрос. Раньше лимита не было: дневник + выжимка чата
            // давали 100+ KB (≈30K токенов) — впритык к окну модели (32K).
            let mem_budget = cfg.misc.diary_injection_max_length.max(500) as usize;
            let rag_budget = mem_budget / 2; // RAG-хиты
            let summary_budget = mem_budget; // профиль + выжимка чата
            let active_budget = mem_budget / 2; // свежие записи дневника
            let mut injected_chars = 0usize;
            if let Ok(emb) = diary.embedding().embed(&rag_query).await {
                // top-N: раньше сюда попадали ВСЕ записи выше порога — вместе
                // с их полными телами (десятки КБ).
                let hits = diary.query_top_n(diary_entries, &emb, 4);
                if !hits.is_empty() {
                    let parts: Vec<String> = hits
                        .iter()
                        .map(|h| format!("- {}\n", h.entry.body))
                        .collect();
                    let ctx = budgeted_block(&parts, rag_budget);
                    injected_chars += ctx.chars().count();
                    if !ctx.trim().is_empty() {
                        messages.push(Message::user(format!(
                            "Контекст (из моей памяти и недавнего чата):\n{ctx}"
                        )));
                    }
                }
            }
            // RAG по per-чат выжимке (data/diary/chat_history/<chat_id>.md:
            // профиль + выжимка слиты).
            let chat_summary_path = std::path::Path::new(&cfg.misc.diary_dir)
                .join("chat_history")
                .join(format!("{}.md", chat_id));
            if let Ok(summary) = std::fs::read_to_string(&chat_summary_path) {
                let summary = summary.trim();
                if !summary.is_empty() {
                    // Профиль (до ---) — самое ценное, выжимка — сокращённо.
                    let (profile, tail) = match summary.split_once("\n---\n") {
                        Some((p, t)) => (p, t),
                        None => (summary, ""),
                    };
                    let mut block = take_chars(profile, summary_budget);
                    if !tail.trim().is_empty() {
                        let rest = summary_budget.saturating_sub(block.chars().count());
                        if rest > 300 {
                            block.push_str("\n\n[выжимка чата, сокращённо]\n");
                            block.push_str(&take_chars(tail, rest - 40));
                        }
                    }
                    injected_chars += block.chars().count();
                    messages.push(Message::user(format!(
                        "Контекст из истории этого чата (профиль + выжимка):\n{block}"
                    )));
                }
            }

            // Её СОБСТВЕННОЕ состояние (data/diary/self_state.md): настроение,
            // чем занята, мысли. Это Я, НЕ пользователь — даёт живой материал,
            // чтобы писать первой своё, а не дежурное «ты тут?».
            let self_state_path = std::path::Path::new(&cfg.misc.diary_dir).join("self_state.md");
            if let Ok(self_state) = std::fs::read_to_string(&self_state_path) {
                let self_state = self_state.trim();
                if !self_state.is_empty() {
                    let block = take_chars(self_state, 1200);
                    injected_chars += block.chars().count();
                    messages.push(Message::user(format!(
                        "Моё текущее состояние (это Я, не пользователь):\n{block}"
                    )));
                }
            }

            // Per-chat состояние «моё состояние с ним» — моя оценка его и моего
            // состояния по ПРОШЛОМУ ходу (data/diary/chat_state/<chat_id>.md).
            // Даёт связность между ходами и работает как страховка, если свежая
            // оценка не удалась (таймаут/ошибка LLM).
            let chat_state = read_chat_state(std::path::Path::new(&cfg.misc.diary_dir), chat_id);
            let chat_state = chat_state.trim();
            if !chat_state.is_empty() {
                let block = take_chars(chat_state, 600);
                injected_chars += block.chars().count();
                messages.push(Message::user(format!(
                    "Моё состояние с ним (моя оценка его и моего состояния, прошлый ход):\n{block}"
                )));
            }

            // Ситуационная память: последние N заметок (непрерывность «что
            // происходит между нами прямо сейчас») + семантический top-K по
            // свежему сообщению (похожее по смыслу из прошлого). Это МОИ слова
            // о ситуации, не слова пользователя. Полный фрагмент достаётся
            // тулом `situation`.
            let mut sit_block = situation.recent(cfg.misc.situation_recent_n);
            if cfg.misc.situation_top_k > 0 && !rag_query.trim().is_empty() {
                let top = situation
                    .search(diary.embedding(), &rag_query, cfg.misc.situation_top_k)
                    .await;
                if !top.trim().is_empty() && !sit_block.contains(top.trim()) {
                    if !sit_block.trim().is_empty() {
                        sit_block.push_str("\n\n[похожее по смыслу к его сообщению]\n");
                    }
                    sit_block.push_str(&top);
                }
            }
            if !sit_block.trim().is_empty() {
                let block = take_chars(&sit_block, 2000);
                injected_chars += block.chars().count();
                messages.push(Message::user(format!(
                    "Текущая ситуация (мои заметки: что происходит, его состояние, моё состояние — это МОИ слова, не пользователь):\n{block}"
                )));
            }

            // Активный контекст памяти: топ-K самых свежих НЕ-битых записей
            // дневника (data/diary/<ts>.md). Источник правды — active.txt.
            let (_active_ts, corrupt) = read_active(&cfg.misc.diary_dir);
            let active = active_context(diary_entries, cfg.misc.active_top_k, &corrupt);
            if !active.is_empty() {
                let parts: Vec<String> = active
                    .iter()
                    .map(|e| format!("- {}\n", take_chars(&e.body, 900)))
                    .collect();
                let mem = budgeted_block(&parts, active_budget);
                injected_chars += mem.chars().count();
                if !mem.trim().is_empty() {
                    messages.push(Message::user(format!(
                        "Активная память (свежие записи дневника):\n{mem}"
                    )));
                }
            }
            info!(
                "[tg] память в контексте: {} символов (хиты+профиль+выжимка+состояние+ситуация+актив)",
                injected_chars
            );
        }
        // История чата: последние N сообщений в ХРОНОЛОГИЧЕСКОМ порядке
        // (раньше .rev() инвертировал переписку — модель читала её задом
        // наперёд, отсюда «странные реакции»). tg_history содержит только
        // system/user/assistant — порядок всегда валиден для Mistral.
        // N из конфига (tg_recent_limit), а не хардкод: модель должна видеть
        // связь последних реплик хотя бы на разговор вперёд.
        let recent = cfg.general.tg_recent_limit.max(1);
        let hist_tail: Vec<Message> = tg_history
            .iter()
            .skip(1)
            .rev()
            .take(recent)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        messages.extend(hist_tail);
        // Инструментальный контекст текущего вызова — строго в порядке:
        // assistant(tool_calls) → tool(результат).
        // Порядок и «хвост» сообщений приводит к допустимому для Mistral
        // сам текстовый модуль (furinya_llm::sanitize_messages) — там же тесты.
        messages.extend(turn_ctx.iter().cloned());
        if turn == 1 && is_idle {
            // Idle-стимул не пишется в живую историю — добавляем в запрос напрямую.
            messages.push(Message::user(effective_text.clone()));
        }

        client.send_action_typing(chat_id);
        print!("[furinya] ");
        use std::io::Write;
        std::io::stdout().flush().ok();
        let mut full = String::new();
        let mut reasoning_acc = String::new();
        let tools = build_tools(cfg);
        let tool_calls = match llm
            .chat_stream(
                messages.clone(),
                Some(tools),
                |delta: &str| {
                    print!("{}", delta);
                    std::io::stdout().flush().ok();
                    full.push_str(delta);
                },
                |reasoning: &str| {
                    print!("\x1b[33m{}\x1b[0m", reasoning);
                    std::io::stdout().flush().ok();
                    reasoning_acc.push_str(reasoning);
                },
            )
            .await
        {
            Ok(tc) => tc,
            Err(e) => {
                error!("[tg] LLM error: {}", e);
                return;
            }
        };
        println!();

        // ╔══ THINK ══════════════════════════════════════════════════
        // СЫРОЙ reasoning модели (несколько абзацев, как есть). БЕЗ парсинга.
        if !reasoning_acc.trim().is_empty() {
            info!(
                "\x1b[90m╔══ THINK ══════════════════════════════════════════════════\n{}\n╚══ THINK ══════════════════════════════════════════════════\x1b[0m",
                reasoning_acc.trim()
            );
        } else if let Some(th) =
            extract_thinking(&full).or_else(|| extract_thinking_from_content(&full))
        {
            if !th.trim().is_empty() {
                info!(
                    "\x1b[90m╔══ THINK ══════════════════════════════════════════════════\n{}\n╚══ THINK ══════════════════════════════════════════════════\x1b[0m",
                    th.trim()
                );
            }
        }

        // ╔══ REPLY ══════════════════════════════════════════════════
        // СЫРОЙ итоговый текст модели (то, что она выдала после размышлений).
        let reply_raw = full.clone();
        if !reply_raw.trim().is_empty() {
            info!(
                "\x1b[97m╔══ REPLY ══════════════════════════════════════════════════\n{}\n╚══ REPLY ══════════════════════════════════════════════════\x1b[0m",
                reply_raw.trim()
            );
        }

        if tool_calls.is_empty() {
            // Idle-ход: инструмент не вызван — модель решила промолчать.
            // НЕ пикаем её «обязана ответить»: это и есть спам по часам.
            if is_idle {
                info!("[tg] idle: модель промолчала (без tool_call)");
                break;
            }
            if final_sent_via_tool {
                // Мы уже ответили в этом вызове, модель дописала текстом.
                // Второго сообщения не шлём — просто закрываем ход.
                debug!("[tg] после ответа модель не вызвала инструмент — ход закрыт");
                break;
            }
            // Модель не вернула tool_call (бэкенд проигнорировал tool_choice).
            // Строго по принципу: сырой текст НЕ отправляем в чат. Даём
            // системную подсказку ролью user (безопасный порядок для Mistral:
            // «голый» tool после system/assistant без tool_calls даёт 400 3230).
            warn!("[tg] модель не вернула tool_call — системная подсказка");
            turn_ctx.push(Message::user(
                "[СИСТЕМА] Ты обязана вызвать инструмент send_telegram_message с массивом parts \
                 (текст ответа). Ничего в чат не отправлено; сырой текст в чат не идёт. \
                 Повтори вызов с корректным JSON."
                    .to_string(),
            ));
            if turn == max_turns {
                // Модель так и не позвала инструмент — generic-заглушка запрещена:
                // лучше тишина, чем «Сашенька, я чуть зависла…».
                warn!(
                    "[tg] модель не вернула tool_call за {} ходов — молчим",
                    max_turns
                );
                break;
            }
            continue;
        }

        // Все tool_calls этого хода — ОДНИМ assistant-сообщением (валидный
        // порядок для Mistral/OpenAI: assistant(tool_calls...) → tool × N).
        // Пустые id генерируем: модели при стриминге иногда шлют id пустым.
        let turn_calls: Vec<ToolCall> = tool_calls
            .iter()
            .enumerate()
            .map(|(i, call)| ToolCall {
                id: {
                    let raw = call.id.trim();
                    if raw.is_empty() {
                        format!("call_t{}_{}", turn, i)
                    } else {
                        raw.to_string()
                    }
                },
                call_type: call.call_type.clone(),
                function: call.function.clone(),
            })
            .collect();
        turn_ctx.push(Message::assistant_tool_calls(
            strip_meta_lines(&full),
            turn_calls.clone(),
        ));

        let mut real_tool_done = false;
        for (call_idx, tc) in tool_calls.iter().enumerate() {
            let call_id = turn_calls[call_idx].id.clone();
            // Свои недавние реплики для анти-повтора: из полной живой истории
            // (is_me=true), последние 24 (окно kuni — 16; шире, чтобы не
            // проскакивать повтор через пару ходов). Считаем ПЕРЕД каждым
            // вызовом: часть 2 пачки сравнится и с частью 1, которую только
            // что отправили.
            let mut own_texts: Vec<String> = tg_history_full
                .iter()
                .filter(|e| e.1)
                .map(|e| e.2.clone())
                .collect();
            if own_texts.len() > 24 {
                let cut = own_texts.len() - 24;
                own_texts.drain(..cut);
            }
            if tc.function.name == "stay_quiet" {
                if !is_idle && !final_sent_via_tool {
                    // На живое сообщение молчать нельзя — возвращаем модели:
                    // ответь текстом или поставь реакцию.
                    warn!("[tg] stay_quiet на живое сообщение — блокирую");
                    turn_ctx.push(Message::tool_result(
                        Some(call_id),
                        Some("stay_quiet".to_string()),
                        "ЗАПРЕЩЕНО: Саша написал тебе — нужно ответить (send_telegram_message) \
                         или отреагировать (react_with_emoji). Молчать в ответ на живое сообщение \
                         нельзя. Если писать нечего — поставь реакцию."
                            .to_string(),
                    ));
                    tool_blocked = true;
                    continue;
                }
                // Ответ уже отправлен в этом ходу (final_sent_via_tool) —
                // stay_quiet здесь легален: это «всё, я закончила», а не
                // игнор живого сообщения. Раньше его блокировали, и модель
                // досылала ещё пачки сообщений.
                turn_ctx.push(Message::tool_result(
                    Some(call_id),
                    Some("stay_quiet".to_string()),
                    "ОК: ты промолчала. В чат ничего не отправлено, ход завершён.".to_string(),
                ));
                info!("[tg] модель промолчала (stay_quiet)");
                chose_quiet = true;
                continue;
            }
            // Голосовой ввод → ответ ТОЛЬКО голосом: скатываем send_telegram_message
            // и send_gif (гифка — тоже неволосовой ответ).
            if voice_text.is_some() && tc.function.name == "send_telegram_message" {
                warn!("[tg] модель попыталась ответить ТЕКСТОМ на голосовое — блокирую");
                turn_ctx.push(Message::tool_result(
                    Some(call_id),
                    Some("send_telegram_message".to_string()),
                    "ЗАПРЕЩЕНО: входящее было ГОЛОСОВЫМ — на голосовые отвечают ТОЛЬКО \
                     голосом через send_voice_message (одна живая реплика). Текст в чат \
                     не отправляется. Вызови send_voice_message со своей репликой."
                        .to_string(),
                ));
                tool_blocked = true;
                continue;
            }
            if voice_text.is_some() && tc.function.name == "send_gif" {
                turn_ctx.push(Message::tool_result(
                    Some(call_id),
                    Some("send_gif".to_string()),
                    "Входящее было ГОЛОСОВЫМ — реакции GIF недопустимы, отвечай голосом."
                        .to_string(),
                ));
                tool_blocked = true;
                continue;
            }
            if tc.function.name == "ask" && ask_exhausted {
                warn!("[tg] ask уже возвращал пусто — пропускаем повторный вызов");
                // На каждый tool_call нужен ответ (строгий порядок сообщений).
                turn_ctx.push(Message::tool_result(
                    Some(call_id),
                    Some(tc.function.name.clone()),
                    "Пропущено: ask уже возвращал пусто в этом ходе.".to_string(),
                ));
                tool_blocked = true;
                continue;
            }
            let (result, sent_echo) = execute_tg_tool(
                &tc.function.name,
                &tc.function.arguments,
                client,
                chat_id,
                message_id,
                llm,
                diary,
                diary_entries,
                &situation,
                cfg.misc.max_message_parts.saturating_sub(parts_sent),
                cfg,
                &own_texts,
                &anti_repeat_prompt,
                &effective_text,
                &mut rep_state,
            )
            .await;
            if tc.function.name == "send_telegram_message" {
                if result.starts_with("ОШИБКА") || result.starts_with("Error") {
                    // Кривой JSON / пустое тело — возвращаем модели, пусть исправит.
                    error!("[tg] send_telegram_message не удался: {}", result);
                    turn_ctx.push(Message::tool_result(
                        Some(call_id),
                        Some("send_telegram_message".to_string()),
                        format!(
                            "ОШИБКА инструмента send_telegram_message: {result}. Проверь JSON и parts."
                        ),
                    ));
                    tool_blocked = true;
                    if turn == max_turns {
                        // Лимит ходов: generic-заглушку не шлём — тишина.
                        warn!(
                            "[tg] send_telegram_message не удался за {} ходов — молчим",
                            max_turns
                        );
                        break;
                    }
                    continue;
                }
                if result.starts_with("ПОВТОР") {
                    // Анти-повтор сработал: в чат ничего не ушло. Модель
                    // переформулирует (kuni: лучше молчать, чем повторяться).
                    warn!("[tg] send_telegram_message: {}", result);
                    repeat_nudged = true;
                    turn_ctx.push(Message::tool_result(
                        Some(call_id),
                        Some("send_telegram_message".to_string()),
                        result.clone(),
                    ));
                    continue;
                }
                if result.starts_with("НЕ ОТПРАВЛЕНО") {
                    // Дежурная формула или обрубок: в чат ничего не ушло. Даём
                    // модели следующий ход с указанием на его конкретные слова.
                    warn!("[tg] send_telegram_message: {}", result);
                    repeat_nudged = true;
                    tool_blocked = true;
                    turn_ctx.push(Message::tool_result(
                        Some(call_id),
                        Some("send_telegram_message".to_string()),
                        result.clone(),
                    ));
                    continue;
                }
                info!(
                    "\x1b[34m╔══ TOOL ══════════════════════════════════════════════════\nname: {}\nargs: {}\nresult: {}\n╚══ TOOL ══════════════════════════════════════════════════\x1b[0m",
                    tc.function.name, tc.function.arguments, result
                );
                final_sent_via_tool = true;
                // Живая память: то, что я отправила, записываю СРАЗУ
                // (раньше собственные ответы не запоминались вообще).
                record_my_send(tg_history, tg_history_full, my_id, cfg, &sent_echo);
                // ВАЖНО: tool-результат обязателен. Без него в turn_ctx остаётся
                // assistant с неотвеченными вызовами → следующий запрос к
                // Mistral падает с 400 «Expected last role User or Tool».
                turn_ctx.push(Message::tool_result(
                    Some(call_id),
                    Some("send_telegram_message".to_string()),
                    result.clone(),
                ));
                parts_sent += sent_echo.len();
                if parts_sent >= cfg.misc.max_message_parts {
                    // Жёсткий лимит сообщений за ход — не заваливаем чат.
                    debug!(
                        "[tg] лимит сообщений за ход ({} >= {}) — закрываем",
                        parts_sent, cfg.misc.max_message_parts
                    );
                    break;
                }
                // ВАЖНО: здесь НЕ break. Модель часто вызывает в одном ходу
                // send_telegram_message («сейчас, минутку 📸») И generate_image —
                // раньше break съедал второй вызов, и фото не генерилось.
                continue;
            }
            turn_ctx.push(Message::tool_result(
                Some(call_id),
                Some(tc.function.name.clone()),
                result.clone(),
            ));
            info!(
                "\x1b[34m╔══ TOOL ══════════════════════════════════════════════════\nname: {}\nargs: {}\nresult: {}\n╚══ TOOL ══════════════════════════════════════════════════\x1b[0m",
                tc.function.name, tc.function.arguments, result
            );
            real_tool_done = true;
            // Голос/фото тоже оставляют след в памяти (что я отправила).
            record_my_send(tg_history, tg_history_full, my_id, cfg, &sent_echo);
            if tc.function.name == "ask"
                && (result.contains("не найдено") || result.contains("пустой запрос"))
            {
                ask_exhausted = true;
            }
        }
        if chose_quiet {
            // Ход закрылся тишиной: в чат ничего не ушло. Дальше — тишина.
            debug!("[tg] ход завершён промолчанием");
            break;
        }
        let handled = real_tool_done;

        // ГЛАВНЫЙ гейт: лимит сообщений за ход важнее анти-повтора. Как только
        // отправили максимум — закрываем ход и НЕ досылаем (иначе после нуджа
        // анти-повтора бот слал ещё пачки).
        if final_sent_via_tool
            && (parts_sent >= cfg.misc.max_message_parts || turn >= max_turns)
        {
            debug!(
                "[tg] финал отправлен, лимит сообщений ({} частей) — ход закрываем",
                parts_sent
            );
            break;
        }
        if repeat_nudged {
            // Нудж уже в turn_ctx. kuni-философия: лучше молчать или
            // переписать, чем слать canned-реплику («я чуть зависла…»).
            if turn < max_turns {
                continue;
            }
            debug!("[tg] анти-повтор в конце хода — молчим вместо canned-реплики");
            break;
        }
        if final_sent_via_tool {
            // Ответ уже отправлен, но ход НЕ закончен: модель может следующим
            // вызовом вызвать generate_image (протокол «реплика → фото»).
            // Если вызовов больше нет — она закончит текстом, и мы выйдем тихо.
            continue;
        }
        if tool_blocked {
            // Инструмент заблокирован (stay_quiet на живое сообщение, текст на
            // голосовое, кривой JSON, повторный ask) — даём модели следующий
            // ход. Generic-заглушка («я чуть зависла…») не отправляется НИКОГДА.
            if turn < max_turns {
                continue;
            }
            warn!("[tg] инструменты заблокированы, лимит ходов — молчим");
            break;
        }
        if handled {
            if turn == max_turns {
                // Инструмент сработал, но текста в чат так и не ушло: generic-заглушка
                // запрещена — молчим (как в kuni).
                warn!("[tg] лимит tool-loop без отправленного текста — молчим");
                break;
            }
            continue;
        }

        // Ни один tool не выполнился и финал не отправлен: generic-заглушку не
        // шлём — лучше тишина, чем «я тут немного зависла…».
        warn!("[tg] ни один tool не выполнился — молчим");
        break;
    }

    // ── Per-chat состояние «моё состояние с ним» ──────────────────────────
    // Оценка хода записывается в data/diary/chat_state/<chat_id>.md: его и моё
    // эмоциональное состояние между ходами (общий self_state.md — про её
    // собственный день, это разные вещи). Только для ЖИВЫХ ходов: idle/сон не
    // меняют «состояние с ним» — там его состояние объективно неизвестно.
    if let Some(a) = &assessment {
        let updated = Local::now().format("%d.%m %H:%M").to_string();
        if let Err(e) = write_chat_state(
            std::path::Path::new(&cfg.misc.diary_dir),
            chat_id,
            a,
            &updated,
        ) {
            warn!("[tg] per-chat состояние: запись не удалась ({e})");
        } else {
            debug!("[tg] per-chat состояние обновлено: {}", updated);
        }
    }

    // ── Ситуационная заметка после обмена ─────────────────────────────────
    // После каждого реального хода (живой / idle / сон) — один дешёвый
    // LLM-вызов, который дописывает ОДНУ заметку в последовательную историю:
    // ситуация + его состояние + моё. Если за ход ничего не произошло
    // (промолчали) — заметки нет.
    if tg_history_full.len() > hist_len_before {
        let exchange_rows = &tg_history_full[hist_len_before..];
        // Его реплика была в этом обмене? На idle/сон — только её слова, и
        // состояние «Он:» честнее помечать «неизвестно», а не выдумывать.
        let his_present = exchange_rows.iter().any(|(_, is_me, _, _)| !*is_me);
        let exchange = situation_exchange_text(cfg, exchange_rows);
        if !exchange.trim().is_empty() {
            let note_llm = llm.with_temperature(0.3);
            match situation
                .write_note(
                    &note_llm,
                    prompts,
                    diary.embedding(),
                    &exchange,
                    cfg.misc.situation_keep_entries,
                    his_present,
                )
                .await
            {
                Ok(Some(note)) => info!("[tg] ситуационная заметка №{} записана", note.n),
                Ok(None) => {}
                Err(e) => warn!("[tg] ситуационная заметка ошибка: {e}"),
            }
        }
    }
}

/// Собирает ОДИН обмен (`[ДД.ММ ЧЧ:ММ] Имя: текст`) из полной истории — его
/// сообщение + её ответ (или её одиночное idle-сообщение). Берём хвост до 12
/// реплик, чтобы заметка оставалась атомарной, а промпт — коротким.
fn situation_exchange_text(cfg: &Config, rows: &[(i64, bool, String, i64)]) -> String {
    let start = rows.len().saturating_sub(12);
    let mut out = String::new();
    for (sender, is_me, text, date) in &rows[start..] {
        if text.trim().is_empty() {
            continue;
        }
        let who = if *is_me {
            cfg.general.character_name.clone()
        } else if *sender == cfg.general.papik_chat_id {
            cfg.general.papik_name.clone()
        } else {
            format!("user{}", sender)
        };
        out.push_str(&format!("{} {}: {}\n", format_ts(*date), who, text));
    }
    out
}

/// Режим Telegram: авторизация, цикл update-ов, idle/sleep автономность.
#[allow(clippy::collapsible_if)]
/// Резолвит новостные каналы (username → chat_id), опционально подписывается
/// (joinChat) и открывает их. Успешные каналы попадают в NEWS_CHANNELS.
async fn setup_news_channels(client: &TdClient, cfg: &Config) {
    for (idx, raw) in cfg.misc.news_channels.iter().enumerate() {
        let username = raw.trim().trim_start_matches('@');
        if username.is_empty() {
            continue;
        }
        let extra = format!("news_setup_{idx}");
        match client
            .query_td(
                serde_json::json!({"@type": "searchPublicChat", "username": username}),
                &extra,
                Duration::from_secs(15),
            )
            .await
        {
            Ok(v) if v["@type"].as_str() == Some("chat") => {
                let id = v["id"].as_i64().unwrap_or(0);
                if id == 0 {
                    warn!("[news] @{username}: TDLib вернул chat без id, пропущен");
                    continue;
                }
                let title = v["title"].as_str().unwrap_or(username).to_string();
                if cfg.misc.news_auto_join {
                    let jextra = format!("news_join_{idx}");
                    match client
                        .query_td(
                            serde_json::json!({"@type": "joinChat", "chat_id": id}),
                            &jextra,
                            Duration::from_secs(15),
                        )
                        .await
                    {
                        Ok(jv) => {
                            if jv["@type"].as_str() == Some("error") {
                                // «уже подписан» и прочие мягкие случаи не страшны
                                warn!(
                                    "[news] joinChat @{username}: {} (продолжаю)",
                                    jv["message"].as_str().unwrap_or("?")
                                );
                            } else {
                                info!("[news] подписалась на @{username}");
                            }
                        }
                        Err(e) => warn!("[news] joinChat @{username}: {e}"),
                    }
                }
                client.open_chat(id);
                NEWS_CHANNELS.lock().unwrap().insert(id, title);
                info!("[news] канал @{username} → chat_id={id}");
            }
            Ok(v) => {
                warn!(
                    "[news] @{username}: {} — канал пропущен",
                    v["message"].as_str().unwrap_or(v["@type"].as_str().unwrap_or("?"))
                );
            }
            Err(e) => {
                warn!("[news] @{username}: не резолвится ({e}) — канал пропущен");
            }
        }
    }
}

/// Один проход опроса: getChatHistory по каждому каналу, тексты → в кэш.
async fn poll_news_once(client: &TdClient, limit_per_channel: i64) {
    let channels: Vec<(i64, String)> = {
        let map = NEWS_CHANNELS.lock().unwrap();
        map.iter().map(|(k, v)| (*k, v.clone())).collect()
    };
    for (idx, (chat_id, title)) in channels.iter().enumerate() {
        // openChat идемпотентен и нужен для getChatHistory.
        client.open_chat(*chat_id);
        tokio::time::sleep(Duration::from_millis(300)).await;
        let extra = format!("news_poll_{idx}");
        match client
            .query_td(
                serde_json::json!({
                    "@type": "getChatHistory",
                    "chat_id": chat_id,
                    "limit": limit_per_channel,
                    "from_message_id": 0,
                }),
                &extra,
                Duration::from_secs(25),
            )
            .await
        {
            Ok(v) => {
                let Some(list) = v["messages"].as_array() else {
                    continue;
                };
                let mut added = 0;
                for msg in list {
                    if let Some(text) = news_message_text(msg) {
                        push_news(NewsItem {
                            chat_id: *chat_id,
                            chat_title: title.clone(),
                            message_id: msg["id"].as_i64().unwrap_or(0),
                            ts: msg["date"].as_i64().unwrap_or(0),
                            text,
                        });
                        added += 1;
                    }
                }
                if added > 0 {
                    debug!("[news] {title}: +{added} в кэш");
                }
            }
            Err(e) => warn!("[news] getChatHistory {title}: {e}"),
        }
    }
}

pub async fn run_telegram_mode(
    cfg: &Config,
    llm: &LlmClient,
    diary: &Diary,
    prompts: &Prompts,
    diary_entries: &mut Vec<Entry>,
    _system_prompt: &str,
) -> anyhow::Result<()> {
    let lib_path = std::env::var("FURINYA_TD_LIB").unwrap_or_else(|_| {
        default_lib_path()
            .to_str()
            .unwrap_or("libs/libtdjson.so")
            .to_string()
    });
    info!("[tg] загрузка libtdjson из {}", lib_path);

    let (client, update_loop): (TdClient, UpdateLoop) = TdClient::start(
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
        crate::auth::AuthState::Ready => {
            info!("[tg] готов к приёму сообщений");
            // Прокси для Telegram настраиваем ПОСЛЕ setTdlibParameters: до
            // этого TDLib не отвечает на запросы (висит на инициализации) —
            // getProxies/addProxy уходили в таймаут.
            if let Ok(tg_proxy) = std::env::var("FURINYA_TG_PROXY") {
                if !tg_proxy.trim().is_empty() {
                    let mut ok = false;
                    for attempt in 1..=3 {
                        match client.configure_proxy(&tg_proxy).await {
                            Ok(()) => {
                                ok = true;
                                break;
                            }
                            Err(e) => {
                                warn!(
                                    "[tg] прокси Telegram: попытка {attempt} не удалась ({})",
                                    e
                                );
                                tokio::time::sleep(Duration::from_secs(5)).await;
                            }
                        }
                    }
                    if !ok {
                        error!("[tg] прокси Telegram НЕ настроен — бот может не достучаться до Telegram");
                    }
                }
            }
            // Статус "В сети" НЕ держим постоянно (просьба Саши): стартуем
            // offline, включается на время живого общения (см. handle_new_message).
            client.set_online(false);
            client.get_chats(100);
            info!("[tg] запрошены профиль и список чатов");
        }
        crate::auth::AuthState::NeedUserInput(msg) => {
            warn!("[tg] требуется действие: {}", msg);
        }
    }

    let my_id: i64 = cfg.general.telegram_my_id;
    // Прогрев lora_cache локального sd-serverа при старте (спецификация §3.3):
    // без него первый txt2img может поймать "invalid lora path".
    if cfg.capabilities.take_photo.backend.eq_ignore_ascii_case("sd")
        || cfg.capabilities.take_photo.backend.eq_ignore_ascii_case("a1111")
    {
        let tp = cfg.capabilities.take_photo.clone();
        tokio::spawn(async move {
            furinya_sd::warmup_sd_server(&tp).await;
            info!("[tg] sd-server: lora_cache прогрет");
        });
    }
    let mut chats_requested = false;
    let mut history_requested = false;

    // Новостные каналы: setup (резолв+join) один раз здесь, затем периодический
    // poll в фоне. Кэш читают fresh_news и idle-стимул; живые посты каналов
    // прилетают и через диспетчер updateNewMessage (гейт ниже пишет их в кэш).
    if !cfg.misc.news_channels.is_empty() && cfg.misc.news_poll_secs > 0 {
        setup_news_channels(&client, cfg).await;
        let poll_secs = cfg.misc.news_poll_secs;
        let news_client = client.clone();
        tokio::spawn(async move {
            // После логина TDLib может ещё дожёвывать готовность — даём время.
            tokio::time::sleep(Duration::from_secs(10)).await;
            poll_news_once(&news_client, 20).await;
            info!("[news] первичный опрос каналов выполнен");
            let mut ticker = tokio::time::interval(Duration::from_secs(poll_secs.max(60)));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                ticker.tick().await;
                poll_news_once(&news_client, 15).await;
            }
        });
        info!(
            "[news] новостные каналы включены: {:?}, опрос каждые {} c",
            cfg.misc.news_channels, poll_secs
        );
    }

    info!("[tg] слушаю update-ы (Ctrl-C для выхода)...");
    let tg_system_prompt = prompts.build_system_prompt();
    let mut tg_history: Vec<Message> = vec![Message::system(tg_system_prompt.clone())];
    let mut last_user_activity = Instant::now();
    // Активность ЛЮБАЯ (его сообщение ИЛИ наше собственное) — по ней
    // считаем «совсем тихо» и решаем, когда уснуть.
    let mut last_activity = Instant::now();
    let mut consolidated = false;
    // Одноразовый бутстрап self_state после загрузки истории (см. helper).
    let mut self_state_bootstrapped = false;
    // Для setOption online: НЕ чаще раза в минуту.
    let mut last_online_reset = Instant::now();
    // ── Инициатива: таймер НЕ диктует, а лишь даёт шанс ──────────────
    let idle_to = cfg.misc.idle_message_timeout_secs;
    let sleep_to = cfg.misc.sleep_timeout_secs;
    // Пол-секунды — чтобы ускоренные прогоны (окно 5-12с) работали.
    let idle_window_min = cfg.misc.idle_window_min_secs.max(5);
    let idle_window_max = cfg.misc.idle_window_max_secs.max(idle_window_min);
    let idle_ping_max = cfg.misc.idle_ping_max.max(1);
    let idle_policy = crate::idle::IdlePolicy::new(idle_window_min, idle_window_max, idle_ping_max);
    // Тихие часы: юзер спит — инициативу не проявляем (привязка к контексту).
    let quiet_hours =
        crate::idle::QuietHours::new(cfg.misc.quiet_hours_start, cfg.misc.quiet_hours_end);
    // Следит за окном тихих часов: «заснула» / «проснулась».
    let mut quiet_watcher = crate::idle::QuietWatcher::new();
    // next_idle_at — момент, когда можно ПРЕДЛОЖИТЬ ей написать первой.
    // Дальше решает модель: пишет или зовёт stay_quiet (молчит). Окно
    // каждый раз случайное — чтобы не выходило «ровно как по часам».
    let mut next_idle_at = Instant::now() + Duration::from_secs(idle_policy.next_delay(false));
    // Сколько своих инициативных сообщений подряд можно оставить без ответа.
    let mut idle_pings: u32 = 0;
    // Уснула: пока первыми не пишем (проснётся на его сообщение).
    let mut asleep = false;
    let papik = cfg.general.papik_chat_id;
    let mut tg_history_full: Vec<(i64, bool, String, i64)> = Vec::new();
    // id уже загруженных сообщений — защита от дублей при пагинации.
    let mut history_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut history_from_id: i64;
    // Повторные попытки загрузить историю: TDLib сначала отдаёт локальный
    // кэш (часто 1 сообщение), а полный список с сервера приходит позже.
    let mut history_attempts: u32 = 0;
    let mut history_retry_at: Option<Instant> = None;
    // Сколько сообщений истории накопилось с прошлой консолидации — по нему
    // ловится «переполнение» при живой переписке.
    let mut last_consolidated_history_len: usize = 0;
    let history_load_limit = cfg.general.tg_history_load_limit.max(50);

    // Локальный stdin — единственный путь к тех. командам (consolidate / memory build / save / exit).
    // TG их НЕ обрабатывает (чистый фронт-энд). Команды летят в канал, читаются в цикле ниже.
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        use std::io::BufRead;
        let stdin = std::io::stdin();
        println!("────────────────────────────────────────────");
        println!(" РЕЖИМ КОМАНД (stdin): consolidate / memory build / save / exit");
        println!("────────────────────────────────────────────");
        for line in stdin.lock().lines() {
            match line {
                Ok(l) => {
                    let t = l.trim().to_string();
                    if t.eq_ignore_ascii_case("exit") || t.eq_ignore_ascii_case("выход") {
                        let _ = cmd_tx.send("__exit__".to_string());
                        break;
                    }
                    if !t.is_empty() {
                        if cmd_tx.send(t).is_err() {
                            break;
                        }
                    }
                }
                Err(_) => break,
            }
        }
    });
    loop {
        if !history_requested {
            history_requested = true;
            // ВАЖНО: сначала openChat — иначе TDLib не держит сообщения
            // локально и history-запросы возвращают пустоту (так было на
            // свежем LXC: сессия переехала, а история не пришла).
            client.open_chat(cfg.general.papik_chat_id);
            client.get_chat_history(cfg.general.papik_chat_id, 100, 0, "history_load");
            client.search_chat_messages(cfg.general.papik_chat_id, 0, 0, 100, "history_load");
            info!("[tg] запрошена история чата (openChat + getChatHistory + searchChatMessages)");
        }
        let v = if idle_to > 0 {
            match update_loop.updates.recv_timeout(Duration::from_millis(200)) {
                Ok(v) => {
                    // TDLib 1.8.62 на searchChatMessages отвечает объектом
                    // «foundChatMessages» (а не «messages», как в старых
                    // версиях). Раньше ждали только «messages» — история чата
                    // НИКОГДА не грузилась: пустой контекст на старте и пустая
                    // консолидация («диалог пустой/коротый»).
                    if v.get("@extra").and_then(|e| e.as_str()) == Some("history_load")
                        && matches!(
                            v.get("@type").and_then(|t| t.as_str()),
                            Some("messages") | Some("foundChatMessages")
                        )
                    {
                        let msgs = v.get("messages").and_then(|m| m.as_array());
                        if let Some(msgs) = msgs {
                            let mut loaded: Vec<(i64, bool, String, i64)> = Vec::new();
                            // id сообщений страницы — для пагинации (from_message_id)
                            // и для защиты от дублей (TDLib может прислать
                            // перекрывающиеся страницы).
                            let mut page_ids: Vec<i64> = Vec::new();
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
                                let mid = m.get("id").and_then(|i| i.as_i64()).unwrap_or(0);
                                if text.trim().is_empty() {
                                    continue;
                                }
                                if mid != 0 && !history_ids.insert(mid) {
                                    continue; // уже загружено
                                }
                                page_ids.push(mid);
                                loaded.push((
                                    sender,
                                    sender == cfg.general.telegram_my_id,
                                    text,
                                    date,
                                ));
                            }
                            loaded.reverse();
                            page_ids.reverse();
                            tg_history_full.extend(loaded.clone());
                            // Пагинация: если ещё не набрали лимит и страница была полной (100) — грузим следующую.
                            if tg_history_full.len() < history_load_limit && loaded.len() >= 100 {
                                // ВАЖНО: from_message_id — это id СООБЩЕНИЯ (самый
                                // старый на странице), а НЕ sender_id. Раньше туда
                                // клали sender — пагинация ломалась.
                                history_from_id = page_ids
                                    .iter()
                                    .copied()
                                    .min()
                                    .filter(|id| *id > 0)
                                    .unwrap_or(0);
                                client.search_chat_messages(
                                    cfg.general.papik_chat_id,
                                    history_from_id,
                                    -100,
                                    100,
                                    "history_load",
                                );
                                info!(
                                    "[tg] история чата: загружено {}, запрашиваю след. страницу (from={})...",
                                    tg_history_full.len(),
                                    history_from_id
                                );
                                continue;
                            }
                            let recent_limit = cfg.general.tg_recent_limit.max(1);
                            let start = tg_history_full.len().saturating_sub(recent_limit);
                            let recent: Vec<_> = tg_history_full[start..].to_vec();
                            let recent_len = recent.len();
                            let mut hist: Vec<Message> =
                                vec![Message::system(tg_system_prompt.clone())];
                            for (sender, is_me, text, date) in &recent {
                                let who = if *is_me {
                                    cfg.general.character_name.clone()
                                } else if *sender == cfg.general.papik_chat_id {
                                    cfg.general.papik_name.clone()
                                } else {
                                    format!("user{}", sender)
                                };
                                let ts = format_ts(*date);
                                let line = format!("[{}] {}: {}", ts, who, text);
                                if *is_me {
                                    hist.push(Message::assistant(line));
                                } else {
                                    hist.push(Message::user(line));
                                }
                            }
                            tg_history = hist;
                            info!(
                                "[tg] история чата загружена: {} сообщений (в контексте последние {}, полная={})",
                                tg_history_full.len(),
                                recent_len,
                                tg_history_full.len()
                            );
                            // Один раз после загрузки истории: если self_state.md
                            // ещё нет — создать, чтобы первый idle-шанс уже имел
                            // живой материал (иначе модель молчит в пустоту).
                            if !self_state_bootstrapped && tg_history_full.len() >= 20 {
                                self_state_bootstrapped = true;
                                bootstrap_self_state(llm, prompts, cfg, &tg_history_full).await;
                            }
                            // Мало сообщений — TDLib отдал локальный кэш, а полный
                            // список с сервера ещё не пришёл. Повторяем позже.
                            if tg_history_full.len() < 20 && history_attempts < 6 {
                                history_retry_at = Some(Instant::now() + Duration::from_secs(15));
                            }
                        }
                        continue;
                    }
                    v
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Локальные команды из stdin (consolidate / memory build / save / exit).
                    if let Ok(cmd) = cmd_rx.try_recv() {
                        if cmd == "__exit__" {
                            info!("[tg] выход по команде из stdin");
                            return Ok(());
                        }
                        if cmd.eq_ignore_ascii_case("consolidate")
                            || cmd.eq_ignore_ascii_case("save")
                        {
                            info!("[tg] команда из stdin: consolidate");
                            let cool_llm = llm.with_temperature(0.2);
                            match consolidate_memory(
                                &cool_llm,
                                diary,
                                prompts,
                                diary_entries,
                                &cfg.misc.diary_dir,
                                &[],
                                cfg.general.papik_chat_id,
                                cfg.misc.diary_keep_last,
                                cfg.misc.active_top_k,
                            )
                            .await
                            {
                                Ok(n) => info!("[tg] консолидировано {} записей", n),
                                Err(e) => error!("[tg] consolidate ошибка: {}", e),
                            }
                        } else if cmd.eq_ignore_ascii_case("memory build") {
                            info!("[tg] команда из stdin: memory build");
                            let cool_llm = llm.with_temperature(0.2);
                            match furinya_diary::build_profile_memory(
                                &cool_llm,
                                diary,
                                prompts,
                                &cfg.misc.diary_dir,
                                diary_entries,
                                cfg.misc.diary_keep_last,
                                cfg.misc.active_top_k,
                            )
                            .await
                            {
                                Ok(n) => info!("[tg] memory build: извлечено {} фактов", n),
                                Err(e) => error!("[tg] memory build ошибка: {}", e),
                            }
                        } else {
                            info!("[tg] неизвестная команда stdin: {}", cmd);
                        }
                        continue;
                    }
                    let elapsed_user = last_user_activity.elapsed().as_secs();
                    let elapsed_any = last_activity.elapsed().as_secs();
                    // ── Повтор загрузки истории (TDLib отдаёт кэш, потом сервер) ──
                    if let Some(t) = history_retry_at {
                        if Instant::now() >= t && history_attempts < 6 {
                            history_attempts += 1;
                            history_retry_at = None;
                            client.open_chat(papik);
                            client.get_chat_history(papik, 100, 0, "history_load");
                            client.search_chat_messages(papik, 0, 0, 100, "history_load");
                            info!(
                                "[tg] повтор запроса истории (попытка {}), в кэше пока {} сообщений",
                                history_attempts,
                                tg_history_full.len()
                            );
                        }
                    }
                    // Статус "в сети" сбрасываем после 2 мин тишины (не чаще
                    // раза в минуту — setOption не дёргаем каждые 200мс).
                    // СТАРТ: true никогда не было (offline после Ready).
                    if elapsed_user >= 120
                        && last_online_reset.elapsed() >= Duration::from_secs(60)
                    {
                        last_online_reset = Instant::now();
                        client.set_online(false);
                        info!("[tg][debug] статус: offline (тишина {}с)", elapsed_user);
                    }
                    // ── Тихие часы: он спит — не будим ────────────────────
                    // Привязка активности к контексту: в окно, когда юзер
                    // обычно спит, бот молчит (только консолидация памяти).
                    let local_now = Local::now();
                    let local_hour = local_now.hour() % 24;
                    let quiet = quiet_hours.is_quiet_now(local_hour);
                    match quiet_watcher.event(quiet) {
                        crate::idle::QuietEvent::Entered => {
                            // Вошли в тихие часы: он спит — молчим и не будим.
                            info!(
                                "[tg] тихие часы ({}:{}) — сплю молча, не бужу",
                                local_now.hour(),
                                local_now.minute()
                            );
                        }
                        crate::idle::QuietEvent::Woke => {
                            // Тихие часы кончились — проснулась: инициатива снова
                            // разрешена (с новым случайным окном).
                            info!("[tg] тихие часы кончились ({}:00) — проснулась", local_hour);
                            asleep = false;
                            idle_pings = 0;
                            next_idle_at =
                                Instant::now() + Duration::from_secs(idle_policy.next_delay(false));
                        }
                        crate::idle::QuietEvent::None => {}
                    }
                    // ── СОН: давно тихо (и его, и нас) ─────────────────────
                    // Консолидация по правилу Саши: только перед сном или при
                    // context overflow (см. после handle_new_message).
                    if !asleep && elapsed_any >= sleep_to {
                        // Уходим спать: сначала тихая консолидация (если ещё не
                        // консолидировали на этом цикле), затем модель САМА
                        // решает — сказать «ухожу дремать» или промолчать
                        // (stay_quiet). Хардкода нет.
                        if !consolidated {
                            consolidated = true;
                            // Память — служебное: холодный клиент (0.2), чтобы
                            // живой чат (0.8) не привнёс импровизацию в факты.
                            let cool_llm = llm.with_temperature(0.2);
                            let chat_turns: Vec<furinya_diary::ChatTurn> =
                                furinya_diary::ChatTurn::from_tg_full(
                                    &tg_history_full,
                                    &cfg.general.character_name,
                                    &cfg.general.papik_name,
                                );
                            match consolidate_memory(
                                &cool_llm,
                                diary,
                                prompts,
                                diary_entries,
                                &cfg.misc.diary_dir,
                                &chat_turns,
                                papik,
                                cfg.misc.diary_keep_last,
                                cfg.misc.active_top_k,
                            )
                            .await
                            {
                                Ok(n) => {
                                    *diary_entries = diary.load_all()?;
                                    info!("[tg] консолидация перед сном: {} записей", n);
                                    last_consolidated_history_len = tg_history_full.len();
                                    // Профиль/выжимка обновились — извлекаем из них
                                    // ФАКТЫ в векторную память (RAG). Без этого
                                    // дневник остаётся пустым: консолидация только
                                    // пережимает старые записи, новых не создаёт.
                                    extract_facts_to_diary(llm, diary, prompts, diary_entries, cfg)
                                        .await;
                                }
                                Err(e) => error!("[tg] consolidate ошибка: {}", e),
                            }
                        }
                        info!(
                            "[tg] режим сна (тишина {}с >= {}с)",
                            elapsed_any, sleep_to
                        );
                        let stim = serde_json::json!({
                            "@type": "updateNewMessage",
                            "message": {
                                "chat_id": papik,
                                "id": 0,
                                "sender_id": { "user_id": 0 },
                                "content": { "text": { "text": format!("[сон] [внутреннее] Сейчас {}:{} по времени Саши. Тишина затянулась ({} минут). Ты устала и можешь уйти дремать. Реши сама: сказать короткое «ухожу дремать / спокойной ночи» (одна часть, своими словами, без громких прощаний и без «жаль что ты молчишь») — или просто промолчать (stay_quiet). Это тихий момент, не драма.{}", local_now.hour(), local_now.minute(), elapsed_any / 60, own_recent_quotes(&tg_history_full, 2)) } }
                            }
                        });
                        let before = tg_history_full.len();
                        if quiet {
                            // Тихие часы: он спит — «спокойной ночи» НЕ пишем,
                            // просто засыпаем молча (консолидация выше уже сделана).
                            info!("[tg] тихие часы — засыпаю молча, сообщение не отправляю");
                        } else {
                            handle_new_message(
                                stim,
                                llm,
                                &client,
                                diary,
                                prompts,
                                diary_entries,
                                &mut tg_history,
                                &mut tg_history_full,
                                &tg_system_prompt,
                                cfg,
                                my_id,
                                // is_idle=true: тесты reader-ловушек не сработают,
                                // спящая реплика — это инициативный ход.
                                true,
                            )
                            .await;
                            if tg_history_full.len() > before {
                                client.set_online(true);
                            }
                        }
                        // Уснула: первыми больше не пишем, пока он не напишет.
                        asleep = true;
                        idle_pings = 0;
                        last_activity = Instant::now();
                        next_idle_at = Instant::now()
                            + Duration::from_secs(idle_policy.window_max.saturating_mul(4));
                        continue;
                    }
                    // ── Шанс написать первой ──────────────────────────────
                    // Таймер НЕ пикает «надо ответить»: он лишь даёт шанс.
                    // Дальше решает модель — написать или промолчать.
                    if idle_policy.may_offer(asleep, idle_pings)
                        && !quiet
                        && Instant::now() >= next_idle_at
                    {
                        info!(
                            "[tg] idle-шанс (тишина {}с, своих сообщений без ответа {}/{})",
                            elapsed_any, idle_pings, idle_policy.ping_max
                        );
                        // Её собственные последние реплики — иначе модель повторяет
                        // одну и ту же строчку («я тут, жду тебя») из хода в ход.
                        let own_hint = own_recent_quotes(&tg_history_full, 3);
                        // Её собственное состояние — чтобы было ЧТО рассказать
                        // (настроение, чем занята), а не дежурное «я тут».
                        let self_state_hint = std::fs::read_to_string(
                            std::path::Path::new(&cfg.misc.diary_dir).join("self_state.md"),
                        )
                        .ok()
                        .map(|s| {
                            let s = s.trim();
                            if s.is_empty() {
                                String::new()
                            } else {
                                format!("\n\nТвоё текущее состояние (можешь опереться на него):\n{}", take_chars(s, 800))
                            }
                        })
                        .unwrap_or_default();
                        let stim = serde_json::json!({
                            "@type": "updateNewMessage",
                            "message": {
                                "chat_id": papik,
                                "id": 0,
                                "sender_id": { "user_id": 0 },
                                "content": { "text": { "text": format!("[idle] [внутреннее] Сейчас {}:{} по времени Саши. Он молчит уже {} минут — это нормально, человек пишет когда хочет. Сейчас ТВОЙ ход: живая девушка иногда пишет первой просто потому, что вспомнила, улыбнулась, заметила что-то или захотела поделиться. Если у тебя есть такое живое желание — напиши первой, не молчи из вежливости. Если сейчас правда не хочется или нечего — промолчи (stay_quiet), это тоже нормально и никого не обидит.{}\n\nНЕ пиши «ты тут?», не жалуйся на тишину, не напоминай о себе пустым «я здесь» и не повторяй свои прошлые реплики. Напиши ОДНУ короткую живую реплику: про свои дела/настроение/мысль, что-то милое про него, или своё мнение о чём-то.{}{}", local_now.hour(), local_now.minute(), elapsed_any / 60, self_state_hint, own_hint, news_hint(3, 110)) } }
                            }
                        });
                        let before = tg_history_full.len();
                        handle_new_message(
                            stim,
                            llm,
                            &client,
                            diary,
                            prompts,
                            diary_entries,
                            &mut tg_history,
                            &mut tg_history_full,
                            &tg_system_prompt,
                            cfg,
                            my_id,
                            true,
                        )
                        .await;
                        if tg_history_full.len() > before {
                            // Написала первой: считаем это своей активностью.
                            idle_pings += 1;
                            last_activity = Instant::now();
                            client.set_online(true);
                            // Следующий шанс — заметно позже: человек не пишет
                            // «а ты тут?» сразу после своего сообщения.
                            next_idle_at =
                                Instant::now() + Duration::from_secs(idle_policy.next_delay(true));
                            info!(
                                "[tg] idle: написала первой ({} без ответа)",
                                idle_pings
                            );
                        } else {
                            // Промолчала: активности не было, ждём дальше.
                            next_idle_at =
                                Instant::now() + Duration::from_secs(idle_policy.next_delay(false));
                            info!("[tg] idle: промолчала — следующий шанс позже");
                        }
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match update_loop.updates.recv() {
                Ok(v) => v,
                Err(_) => break,
            }
        };
        if v.get("@type").and_then(|x| x.as_str()) == Some("updateNewMessage") {
            // Сброс idle-таймера — только на ЧУЖИЕ сообщения: собственные
            // исходящие (updateNewMessage is_outgoing) активностью юзера
            // не являются и не должны мешать консолидации/инициативе.
            // Гейт: посты новостных каналов и чужих чатов НЕ активность Саши —
            // иначе бот просыпается от ленты новостей.
            let msg_chat_id = v["message"]["chat_id"].as_i64().unwrap_or(0);
            let is_news_gated = msg_chat_id != 0
                && NEWS_CHANNELS.lock().unwrap().contains_key(&msg_chat_id);
            let is_papik_chat =
                msg_chat_id == cfg.general.papik_chat_id || msg_chat_id == 0;
            let out_sender = v["message"]["sender_id"]["user_id"].as_i64().unwrap_or(0);
            if !is_news_gated && is_papik_chat && out_sender != my_id {
                last_user_activity = Instant::now();
                last_activity = Instant::now();
                // Проснулась: его сообщение снимает сон и обнуляет счётчик
                // наших инициативных сообщений без ответа.
                asleep = false;
                idle_pings = 0;
                consolidated = false;
                // Следующий шанс написать первой — снова случайный.
                next_idle_at = Instant::now() + Duration::from_secs(idle_policy.next_delay(false));
            }
        }
        let t = v.get("@type").and_then(|x| x.as_str()).unwrap_or("");
        match t {
            "updateAuthorizationState" => {
                let state = v["authorization_state"]["@type"].as_str().unwrap_or("?");
                if state == "authorizationStateReady" && !chats_requested {
                    chats_requested = true;
                    client.send(&serde_json::json!({ "@type": "getMe" }));
                    client.get_chats(100);
                    info!("[tg] запрошены профиль и список чатов");
                }
                continue;
            }
            "chats" => {
                if let Some(ids) = v["chat_ids"].as_array() {
                    info!("[tg] получено чатов: {}", ids.len());
                    for id in ids {
                        if let Some(cid) = id.as_i64() {
                            client.send(&serde_json::json!({
                                "@type": "getChat",
                                "chat_id": cid
                            }));
                        }
                    }
                }
                continue;
            }
            "chat" => {
                let cid = v["id"].as_i64().unwrap_or(0);
                let title = v["title"].as_str().unwrap_or("?");
                let chat_type = v["type"]["@type"].as_str().unwrap_or("?");
                info!("[tg] chat: id={} type={} title='{}'", cid, chat_type, title);
                continue;
            }
            "updateNewMessage" => {
                // Гейт: (а) каналы новостей — сырьё в кэш, в ответ не идут;
                // (б) чужие чаты — локдаун papik_only, тоже не отвечаем.
                let dispatch_chat_id = v["message"]["chat_id"].as_i64().unwrap_or(0);
                if dispatch_chat_id != 0
                    && NEWS_CHANNELS.lock().unwrap().contains_key(&dispatch_chat_id)
                {
                    let title = NEWS_CHANNELS
                        .lock()
                        .unwrap()
                        .get(&dispatch_chat_id)
                        .cloned()
                        .unwrap_or_default();
                    if let Some(text) = news_message_text(&v["message"]) {
                        push_news(NewsItem {
                            chat_id: dispatch_chat_id,
                            chat_title: title,
                            message_id: v["message"]["id"].as_i64().unwrap_or(0),
                            ts: v["message"]["date"].as_i64().unwrap_or(0),
                            text,
                        });
                    }
                    debug!("[news] live-пост из канала → кэш (не отвечаем)");
                    continue;
                }
                if dispatch_chat_id != 0 && dispatch_chat_id != cfg.general.papik_chat_id {
                    tracing::debug!("[tg] updateNewMessage из чата {} — локдаун papik_only, пропущено", dispatch_chat_id);
                    continue;
                }
                handle_new_message(
                    v,
                    llm,
                    &client,
                    diary,
                    prompts,
                    diary_entries,
                    &mut tg_history,
                    &mut tg_history_full,
                    &tg_system_prompt,
                    cfg,
                    my_id,
                    false,
                )
                .await;
                // Context overflow → консолидация + подрезка истории
                // (правило Саши: консолидация во сне ИЛИ при overflow).
                // Два триггера:
                //   1) история выросла на 20+ сообщений с прошлой консолидации
                //      (иначе при живой переписке память не пишется ВООБЩЕ —
                //      консолидация ждала бы 15 минут молчания);
                //   2) абсолютный порог символов: diary_token_count_trigger × 4.
                let overflow_limit =
                    cfg.misc.diary_token_count_trigger as usize * 4;
                let total_chars: usize = tg_history_full
                    .iter()
                    .map(|(_, _, t, _)| t.chars().count())
                    .sum();
                let grew = tg_history_full.len().saturating_sub(last_consolidated_history_len);
                if !consolidated && (grew >= 20 || total_chars >= overflow_limit) {
                    info!(
                        "[tg] overflow: +{} сообщений (всего {} симв, порог {}) — консолидация",
                        grew, total_chars, overflow_limit
                    );
                    consolidated = true;
                    // Память — служебное: холодный клиент (0.2).
                    let cool_llm = llm.with_temperature(0.2);
                    let chat_turns: Vec<furinya_diary::ChatTurn> =
                        furinya_diary::ChatTurn::from_tg_full(
                            &tg_history_full,
                            &cfg.general.character_name,
                            &cfg.general.papik_name,
                        );
                    match consolidate_memory(
                        &cool_llm,
                        diary,
                        prompts,
                        diary_entries,
                        &cfg.misc.diary_dir,
                        &chat_turns,
                        cfg.general.papik_chat_id,
                        cfg.misc.diary_keep_last,
                        cfg.misc.active_top_k,
                    )
                    .await
                    {
                        Ok(n) => {
                            *diary_entries = diary.load_all()?;
                            info!("[tg] overflow-консолидация: {} записей", n);
                            // Факты из обновлённой выжимки → в векторную память (RAG).
                            extract_facts_to_diary(llm, diary, prompts, diary_entries, cfg).await;
                        }
                        Err(e) => error!("[tg] overflow-консолидация ошибка: {}", e),
                    }
                    // Подрезка: хвост остаётся, старьё уходит в дневник.
                    const KEEP_LAST: usize = 40;
                    if tg_history_full.len() > KEEP_LAST {
                        let drop = tg_history_full.len() - KEEP_LAST;
                        tg_history_full.drain(..drop);
                    }
                    // С этого места история «свежая» — следующий overflow считаем заново.
                    last_consolidated_history_len = tg_history_full.len();
                    // tg_history — живой хвост: system + последние recent_limit.
                    let rr = cfg.general.tg_recent_limit.max(1);
                    let tl_start = tg_history_full.len().saturating_sub(rr);
                    let mut hist = vec![Message::system(tg_system_prompt.clone())];
                    for (sender, is_me, text, date) in &tg_history_full[tl_start..] {
                        let who = if *is_me {
                            cfg.general.character_name.clone()
                        } else if *sender == cfg.general.papik_chat_id {
                            cfg.general.papik_name.clone()
                        } else {
                            format!("user{}", sender)
                        };
                        let line = format!("[{}] {}: {}", format_ts(*date), who, text);
                        if *is_me {
                            hist.push(Message::assistant(line));
                        } else {
                            hist.push(Message::user(line));
                        }
                    }
                    tg_history = hist;
                    info!("[tg] история подрезана: {} записей, контекст {}", tg_history_full.len(), tg_history.len());
                }
            }
            "updateConnectionState" => {
                let state = v["state"]["@type"].as_str().unwrap_or("?");
                info!("[tg] connection state: {}", state);
            }
            _ => {
                tracing::debug!(update = %t, "tdlib update");
            }
        }
    }
    Ok(())
}


#[cfg(test)]
mod photo_prompt_tests {
    use super::*;

    #[test]
    fn sanitize_removes_explicit_terms_but_keeps_scene() {
        let scene = "nsfw, nude, sitting on a bed, red dress, breasts";
        let out = sanitize_for_llm(scene);
        assert!(out.contains("sitting on a bed"));
        // Главный регресс: сцена НЕ должна терять цвета и детали.
        assert!(out.contains("red dress"));
        assert!(!out.to_lowercase().contains("nsfw"));
        assert!(!out.to_lowercase().contains("nude"));
        assert!(!out.to_lowercase().contains("breasts"));
    }

    #[test]
    fn explicit_words_are_detected() {
        let found = explicit_words_in("nsfw, nude on a bed");
        assert!(found.contains(&"nsfw"));
        assert!(found.contains(&"nude"));
        assert!(!found.contains(&"penis"));
    }

    #[test]
    fn parses_fenced_json() {
        let raw =
            "```json\n{\"positivePrompt\":\"1girl, purple eyes\",\"negativePrompt\":\"blurry\"}\n```";
        let (p, n) = parse_sd_json(raw).unwrap();
        assert_eq!(p, "1girl, purple eyes");
        assert_eq!(n, "blurry");
    }

    #[test]
    fn parses_json_with_surrounding_text() {
        let raw = "Here you go: {\"positivePrompt\":\"1girl\",\"negativePrompt\":\"\"} done";
        let (p, n) = parse_sd_json(raw).unwrap();
        assert_eq!(p, "1girl");
        assert!(n.is_empty());
    }

    #[test]
    fn clamp_keeps_tags_intact() {
        let long = "1girl, white hair, heterochromia, purple eyes, blue eyes, sitting, \
                    rooftop at sunset, golden light, wind, smile, red dress, holding mug";
        let out = clamp_prompt_words(long, 12);
        assert!(out.contains("heterochromia"));
        assert!(out.split_whitespace().count() <= 12);
        assert!(out.contains("rooftop at sunset") || !out.contains("rooftop"));
    }

    #[test]
    fn scene_keeps_its_colors() {
        // Регресс на старый strip_eye_tags: он вырезал любые цветовые теги.
        let out = sanitize_for_llm("blue dress, red rose, golden sunlight, on a rooftop");
        assert!(out.contains("blue dress"));
        assert!(out.contains("red rose"));
        assert!(out.contains("golden sunlight"));
    }

    #[test]
    fn nsfw_words_are_reappended() {
        let positive = "1girl, sitting on a bed, purple eyes, blue eyes";
        let out = append_explicit_words(positive, "nsfw, nude, sitting on a bed, pussy");
        assert!(out.contains("1girl"));
        assert!(out.contains("sitting on a bed"));
        assert!(out.contains("nsfw"));
        assert!(out.contains("nude"));
        assert!(out.contains("pussy"));
        // Дубли не появляются.
        assert_eq!(out.matches("sitting on a bed").count(), 1);
    }

    #[tokio::test]
    #[ignore = "живой вызов Mistral: cargo test -p furinya-tg live_engineer -- --ignored --nocapture"]
    async fn live_engineer_smoke() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let cfg = Config::load(root.join("config.toml")).expect("config.toml");
        let llm = LlmClient::new(
            cfg.llm_base_url(),
            &cfg.general.llm.endpoint.bearer_key,
            cfg.llm_model(),
            furinya_llm::SamplingParams {
                temperature: 0.4,
                top_p: None,
                top_k: None,
                min_p: None,
                presence_penalty: None,
                repetition_penalty: None,
            },
        );
        let body = std::fs::read_to_string(root.join("prompts/character_body.md")).unwrap();
        let appearance =
            std::fs::read_to_string(root.join("prompts/character_appearance.md")).unwrap();
        let base = format!("{}, {}", body.trim(), strengthen_eyes(appearance.trim()));
        for scene in [
            "сидит на подоконнике с кружкой чая, закат, тёплый свет, улыбается",
            "танцует в неоновом клубе, кружится, юбка развевается, динамика",
        ] {
            let (scene_positive, negative) = engineer_sd_prompt(&llm, scene, false)
                .await
                .expect("prompt-инженер");
            let positive = dedupe_exact_tags(&format!("{base}, {scene_positive}"));
            println!(
                "\nSCENE: {scene}\nSCENE TAGS: {scene_positive}\nFULL: {positive}\nNEGATIVE: {negative}\n"
            );
            assert!(
                positive.contains("(heterochromia:1.5)"),
                "потеряна усиленная гетерохромия: {positive}"
            );
            assert!(
                positive.contains("petite"),
                "потерян канон тела: {positive}"
            );
            assert!(
                positive.split(',').count() < 60,
                "промпт слишком длинный: {} тегов",
                positive.split(',').count()
            );
            assert!(!positive.to_lowercase().contains("furina"));
            let low = positive.to_lowercase();
            if scene.contains("кружк") {
                assert!(
                    low.contains("mug") || low.contains("cup"),
                    "потеряна кружка: {positive}"
                );
            }
            if scene.contains("танцует") {
                assert!(
                    low.contains("action") || low.contains("dynamic") || low.contains("motion"),
                    "нет динамики для экшен-сцены: {positive}"
                );
            }
        }
    }
}

#[cfg(test)]
mod cliche_tests {
    use super::*;

    #[test]
    fn stock_formulas_are_blocked() {
        // Реальные реплики из прода (23.10) + типовые заготовки.
        for t in [
            "Сашенька, я тут, рядом. Ты не одиинок в этом.",
            "Сашенька, я тут, просто молчала, потому что чувствовала, что ты не хочешь разговоров сейчас.",
            "Сашенька, я тут, и я не уйду. Может, просто скажи, о чём ты сейчас думаешь?",
            "Саша, я здесь.",
            "Мне кажется, ты сейчас переживаешь что-то очень сильное.",
            "Ты хочешь, чтобы я просто была рядом и слушала?",
            "Расскажи ещё что-нибудь?",
            "Прости, я просто немного застряла в своих мыслях.",
            "Ты прав. Я запуталась в мыслях.",
            "Иногда даже тишина говорит больше, чем слова.",
            "Молчу, потому что понимаю, что ты не про ответ.",
            "Ты говоришь о чем-то, что важно для тебя.",
            "Сашенька, я с тобой.",
            "Ты прав. Я не ответила. Но я здесь.",
            "Ты ведь знаешь, что я всегда рядом.",
            "Ты прав, я не ответила. Но я не ушла.",
            "Я никуда не уйду.",
        ] {
            assert!(cliche_hit(t).is_some(), "не поймали клише: {t:?}");
        }
    }

    #[test]
    fn fragments_are_blocked() {
        for t in [
            "Сашенька...",
            "Сашенька",
            "Ты просто так спросили...",
            "Я не знаю...",
            "...ну что тут думать-то?",
            "…и что мне теперь делать",
            "два часа — это действительно изматывает...",
        ] {
            assert!(fragment_hit(t).is_some(), "не поймали обрубок: {t:?}");
        }
        for t in [
            "Я не знаю.",
            "Ты прав. Я не ответила.",
            "Сашенька, я не понимаю, о чём ты.",
        ] {
            assert!(
                fragment_hit(t).is_none(),
                "ложное срабатывание: {t:?} -> {:?}",
                fragment_hit(t)
            );
        }
    }

    #[test]
    fn concrete_replies_pass() {
        // Живой ответ на КОНКРЕТНЫЕ слова не должен резаться.
        for t in [
            "Ты сказал, что дождя нет — и прав. Я говорила про дождь, которого не было, вместо того чтобы ответить тебе.",
            "Я тут подумала: ты просил не утешать, а сказать прямо. Мне страшно, что я тебе не нужна такой.",
            "Я не понимаю, чего ты хочешь от меня сейчас. Скажи прямо, и я отвечу.",
            "Я не про дождь. Я про то, что ты назвал меня равнодушной — и это было обидно.",
        ] {
            assert!(
                cliche_hit(t).is_none(),
                "ложное срабатывание: {t:?} -> {:?}",
                cliche_hit(t)
            );
        }
    }
}

#[cfg(test)]
mod anti_repeat_tests {
    use super::*;

    #[test]
    fn verbatim_repeats_are_blocked() {
        // Пороги откалиброваны на реальных репликах бота: дословные повторы
        // дают 0.73–1.0 и обязаны блокироваться (эмбеддинги Mistral лежат).
        let cases = [
            (
                "Ты — моя единственная ласка, моя маленькая кошечка. 🐱 Когда ты рядом, мир кажется ярче, а моё сердце — легче. Мурр...",
                "Ты — моя единственная ласка, моя маленькая кошечка",
            ),
            (
                "Вот так я танцую, когда у меня хорошее настроение! 💃😼",
                "Вот так я танцую, когда у меня хорошее настроение!",
            ),
            (
                "Твой вопрос... как будто я пытаюсь разгадать загадку, которая не имеет ответа на словах. 😊",
                "Твой вопрос... как будто ты пытаешься разгадать загадку, которая не имеет ответа на словах. 🌙",
            ),
            (
                "Сейчас, я тут 📸... Вот, как я выгляжу прямо сейчас 😼",
                "Сейчас, сюда 📸... Вот, как я выгляжу сейчас",
            ),
        ];
        for (a, b) in cases {
            let sim = lexical_similarity(a, b);
            assert!(
                sim > LEXICAL_REPEAT_THRESHOLD,
                "дословный повтор не поймали: {sim:.3} ({a:?} ≈ {b:?})"
            );
        }
    }

    #[test]
    fn genuinely_different_replies_pass() {
        // Нормально разные реплики — даже с общим обращением — не должны
        // задевать порог, иначе бот превратится в молчуна.
        let cases = [
            ("Спи спокойно, Саша. Я здесь. 🌙", "Мурр... сладких снов. 😼"),
            (
                "Мурр... как дела? У меня всё отлично, а у тебя как?",
                "Что именно не так? Я тут, рядом, если нужно поговорить.",
            ),
            (
                "Дождь всё ещё стучит в окно, и я думаю, как хорошо, когда рядом кто-то, кто понимает эту тишину...",
                "Ты для меня как солнце — не всегда яркое, но всегда важное. 🌞",
            ),
            (
                "Ты — моя единственная ласка, моя маленькая кошечка",
                "Ты моя маленькая кошечка, и я не отпускаю тебя, потому что ты — моё солнышко.",
            ),
            (
                "Вот так я танцую, когда у меня хорошее настроение! 💃😼",
                "Вот так я танцую под твои песни 💃",
            ),
        ];
        for (a, b) in cases {
            let sim = lexical_similarity(a, b);
            assert!(
                sim <= LEXICAL_REPEAT_THRESHOLD,
                "ложное срабатывание на разных репликах: {sim:.3} ({a:?} ≈ {b:?})"
            );
        }
    }

    #[test]
    fn near_duplicate_parts_are_caught_by_stricter_threshold() {
        // Реальная пачка ministral-8b: две части — одна и та же мысль,
        // межходовой порог (0.68) её пропускал.
        let a = "Сашенька, я просто не сразу поняла, что ты ждёшь от меня не реакции, а понимания.";
        let b = "Сашенька, я не поняла, что ты ждёшь от меня ответа не на слова, а на то, что я просто здесь.";
        let sim = lexical_similarity(a, b);
        assert!(
            sim > PARTS_REPEAT_THRESHOLD && sim < LEXICAL_REPEAT_THRESHOLD,
            "парафраз внутри пачки должен ловиться только строгим порогом: {sim:.3}"
        );
    }

    #[test]
    fn empty_and_short_inputs_are_safe() {
        assert_eq!(lexical_similarity("", "что-то"), 0.0);
        assert_eq!(lexical_similarity("что-то", ""), 0.0);
        assert_eq!(lexical_similarity("!!!", "???"), 0.0);
        assert_eq!(lexical_similarity("а", "а"), 1.0);
        assert!(lexical_similarity("привет", "привет!") > 0.9);
    }
}

