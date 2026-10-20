//! Агент Telegram: цикл приёма update-ов, агентный loop ответа,
//! консолидация памяти при сне/idle, загрузка истории чата.
//!
//! Весь функционал общения с Telegram живёт здесь (а не в main.rs).

use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant};

use furinya_core::config::Config;
use furinya_core::prompts::Prompts;
use furinya_diary::{
    Diary, Entry, active_context, consolidate_memory, read_active,
};
use furinya_llm::{
    LlmClient, Message, ToolCall, build_tools, extract_thinking, extract_thinking_from_content,
};
use furinya_vision::describe_photo;
use furinya_audio::{synthesize_speech_openai, synthesize_fish_voice, transcribe_audio};
use furinya_sd::generate_image;
use chrono::{Local, Timelike, Utc};
use rand::Rng;
use serde_json::Value;
use tracing::{debug, error, info, warn};

use crate::{TdClient, UpdateLoop, auth, default_lib_path};

/// Дробит ответ модели на части для отправки несколькими сообщениями.
fn split_message(text: &str, max_parts: usize) -> Vec<String> {
    let max_parts = max_parts.clamp(1, 10);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return vec![String::new()];
    }
    let parts: Vec<String> = trimmed
        .split("\n\n")
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() >= 2 {
        parts.into_iter().take(max_parts).collect()
    } else {
        vec![trimmed.to_string()]
    }
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
    diary: &Diary,
    diary_entries: &mut Vec<Entry>,
    cfg: &Config,
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
            // Живая очередь сообщений (spec от Саши: минимум 1, максимум 6;
            // лучше меньше, но одно крупное сообщение тоже ок).
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
            let parts: Vec<_> = merged.into_iter().take(6).collect();
            let mut rng = rand::thread_rng();
            for part in &parts {
                // Реалистичный ввод текста: typing-action + пауза по длине
                // (было: фиксированные 400ms → выглядели механически-быстро).
                let chars = part.chars().count() as u64;
                let typing_ms = (600 + 45 * chars).clamp(800, 3200)
                    .saturating_add(rng.gen_range(0..500));
                client.send_action_typing(chat_id);
                tokio::time::sleep(Duration::from_millis(typing_ms)).await;
                // Живость: в ~20% случаев отправляем с одной опечаткой и
                // тут же редактируем на верный текст («перепечатала»).
                // В контекст запись идёт КОРРЕКТНЫМ текстом (parts не тащим).
                let to_send = if rng.gen_bool(0.20) {
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
                    let hits = diary.query_top_n(diary_entries, &vec, 4);
                    if hits.is_empty() {
                        "Дневник: релевантных воспоминаний не найдено.".to_string()
                    } else {
                        let mut out = String::from("Дневник (топ релевантных):\n");
                        for h in &hits {
                            let preview: String = h
                                .entry
                                .body
                                .chars()
                                .take(500)
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
            let prompt = v
                .get("prompt")
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .to_string();
            if prompt.trim().is_empty() {
                return (
                    "ОШИБКА: generate_image получил пустой prompt.".to_string(),
                    Vec::new(),
                );
            }
            let negative_prompt = v
                .get("negative_prompt")
                .and_then(|p| p.as_str())
                .unwrap_or("")
                .to_string();
            // NSFW — только явным флагом от модели (特殊 perms: приватный чат с
            // Сашей). Если模型 забыл указать, но промпт открытый — верим промпту.
            let mut nsfw = v.get("nsfw").and_then(|p| p.as_bool()).unwrap_or(false);
            let p_lower = prompt.to_lowercase();
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
            // Канон внешки: теги из prompts/character_appearance.md ВСЕГДА
            // идут в начало промпта. Раньше модель сочиняла вид каждый раз
            // заново → картинки были "всратые и очень разные". Теперь стиль и
            // персонаж постоянны, LLM добавляет только сцену/позу/настроение.
            let canon_tags = std::fs::read_to_string("prompts/character_appearance.md")
                .unwrap_or_default()
                .trim()
                .trim_end_matches(['.', ','])
                .to_string();
            let prompt = prompt.trim().to_string();
            let prompt = if canon_tags.is_empty() {
                prompt
            } else {
                format!("{canon_tags}, {prompt}")
            };
            // Дефолтный негатив: модель часто не заполняет negative_prompt, а
            // SD1.5-воркеры Horde без него выдают мусор (артифакты, лишние
            // пальцы, watermark). " ### " — Horde-синтаксис входа негатива.
            let negative_prompt = if negative_prompt.trim().is_empty() {
                "worst quality, low quality, jpeg artifacts, watermark, text, signature, \
                 blurry, deformed, bad anatomy, extra fingers, extra limbs, mutated, ugly"
                    .to_string()
            } else {
                negative_prompt
            };
            // Куда сохраняем сгенерированный PNG.
            let out_dir = std::env::temp_dir();
            let fname = format!(
                "furinya_gen_{}.png",
                Utc::now().format("%Y%m%d_%H%M%S")
            );
            let out_path = out_dir.join(&fname);
            info!("[tg] generate_image: промпт={:?} → бэкенд {} (nsfw={nsfw})", prompt, cfg.capabilities.take_photo.backend);
            client.send_action_typing(chat_id);
            match generate_image(
                &cfg.capabilities.take_photo,
                &prompt,
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
                        vec![format!("[я отправила ФОТО (промпт): {prompt}]")],
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
    _prompts: &Prompts,
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
                                    temperature: cfg.misc.llm_temperature,
                                    top_p: cfg.misc.llm_top_p.parse().ok(),
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

    // Технические команды (consolidate/save/exit) НЕ обрабатываются в TG —
    // только в локальный stdin (см. main.rs). В TG игнорируем их полностью.
    //
    // Автоматический 👍 УБРАН: реакция «на каждое сообщение» была механической
    // и обесценивалась. Модель сама решает через react_with_emoji (см.
    // описание инструмента) — ставить ли реакцию и какую по настроению.

    let max_turns = 8;
    let mut ask_exhausted = false;
    let mut final_sent_via_tool = false;
    // Сколько сообщений отправили за весь вызов (максимум 2) — чтобы после
    // ответа модель не заваливала чат.
    let mut message_sends = 0;
    // Модель САМА выбрала промолчать (stay_quiet) — ход закрывается тишиной.
    let mut chose_quiet = false;
    // Инструментальный диалог ЭТОГО вызова (assistant tool_calls → tool-результаты).
    // Ничего с role=tool не должно попадать в tg_history: «осиротевший» tool
    // без предшествующего assistant.tool_calls роняет Mistral ошибкой
    // 400 "Unexpected role 'tool' after role 'system'" (код 3230).
    let mut turn_ctx: Vec<Message> = Vec::new();
    for turn in 1..=max_turns {
        let mut messages = vec![Message::system(tg_system_prompt.to_string())];
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
                "[tg] память в контексте: {} символов (хиты+профиль+выжимка+актив)",
                injected_chars
            );
        }
        // История чата: последние 8 сообщений в ХРОНОЛОГИЧЕСКОМ порядке
        // (раньше .rev() инвертировал переписку — модель читала её задом
        // наперёд, отсюда «странные реакции»). tg_history содержит только
        // system/user/assistant — порядок всегда валиден для Mistral.
        let hist_tail: Vec<Message> = tg_history
            .iter()
            .skip(1)
            .rev()
            .take(8)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        messages.extend(hist_tail);
        // Инструментальный контекст текущего вызова — строго в порядке:
        // assistant(tool_calls) → tool(результат).
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
        let tools = build_tools();
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
                let reply = "Сашенька, я чуть зависла... но я с тобой! 🥰 Расскажи ещё что-нибудь?";
                for chunk in split_message(reply, cfg.misc.max_message_parts) {
                    client.send_action_typing(chat_id);
                    client.send_text(chat_id, &chunk);
                    tokio::time::sleep(Duration::from_millis(400)).await;
                }
                remember_in_history(tg_history, Message::assistant(reply.to_string()));
                record_my_send(tg_history, tg_history_full, my_id, cfg, &[reply.to_string()]);
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
            if tc.function.name == "stay_quiet" {
                if !is_idle {
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
                    continue;
                }
                // Idle-ход: модель САМА выбрала промолчать. В чат ничего не уходит.
                // Остальные вызовы этого хода (если есть) обрабатываем как обычно —
                // молчание не должно съесть уже подготовленное сообщение.
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
                continue;
            }
            if voice_text.is_some() && tc.function.name == "send_gif" {
                turn_ctx.push(Message::tool_result(
                    Some(call_id),
                    Some("send_gif".to_string()),
                    "Входящее было ГОЛОСОВЫМ — реакции GIF недопустимы, отвечай голосом."
                        .to_string(),
                ));
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
                continue;
            }
            let (result, sent_echo) = execute_tg_tool(
                &tc.function.name,
                &tc.function.arguments,
                client,
                chat_id,
                message_id,
                diary,
                diary_entries,
                cfg,
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
                    if turn == max_turns {
                        let reply =
                            "Сашенька, я чуть зависла... но я с тобой! 🥰 Расскажи ещё что-нибудь?";
                        // Голосовой ввод → фолбэк-текст НЕ отправляем.
                        if voice_text.is_none() {
                            for chunk in split_message(reply, cfg.misc.max_message_parts) {
                                client.send_action_typing(chat_id);
                                client.send_text(chat_id, &chunk);
                                tokio::time::sleep(Duration::from_millis(400)).await;
                            }
                            remember_in_history(tg_history, Message::assistant(reply.to_string()));
                            record_my_send(tg_history, tg_history_full, my_id, cfg, &[reply.to_string()]);
                        } else {
                            warn!("[tg] фолбэк-текст НЕ отправлен: голосовой ввод требует голос");
                        }
                        break;
                    }
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
                message_sends += 1;
                if message_sends >= 2 {
                    // Больше двух сообщений за вызов не шлём.
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

        if final_sent_via_tool {
            // Ответ уже отправлен, но ход НЕ закончен: модель может следующим
            // вызовом вызвать generate_image (протокол «реплика → фото»).
            // Если вызовов больше нет — она закончит текстом, и мы выйдем тихо.
            if message_sends >= 2 || turn >= max_turns {
                debug!("[tg] финал отправлен, ход закрываем");
                break;
            }
            continue;
        }
        if handled {
            if turn == max_turns {
                if is_idle {
                    // Idle: canned-реплика — это ровно тот «по часам» спам. Молчим.
                    warn!("[tg] idle: лимит tool-loop — промолчали вместо canned-реплики");
                    break;
                }
                if final_sent_via_tool {
                    // Ответ уже отправлен — canned-реплика не нужна.
                    debug!("[tg] лимит tool-loop, но ответ уже отправлен — молча закрываем");
                    break;
                }
                warn!(
                    "[tg] достигнут лимит tool-loop ({}), отправляю fallback-ответ",
                    max_turns
                );
                let reply = "Сашенька, я чуть зависла на своих мыслях... но я с тобой! 🥰 Расскажи ещё что-нибудь?";
                let reply = reply.trim();
                // Голосовой ввод → фолбэк-текст НЕ отправляем (правило
                // «голосовое → ответ только голосом»).
                if voice_text.is_none() {
                    let chunks = split_message(reply, cfg.misc.max_message_parts);
                    for (i, chunk) in chunks.iter().enumerate() {
                        if i > 0 {
                            let wpm = cfg.misc.typing_simulation_min_wpm
                                + (cfg.misc.typing_simulation_max_wpm
                                    - cfg.misc.typing_simulation_min_wpm)
                                    / 2;
                            let words = chunk.split_whitespace().count().max(1);
                            let secs = (words as f64 / wpm as f64 * 60.0).clamp(0.3, 3.0);
                            tokio::time::sleep(Duration::from_secs_f64(secs)).await;
                        }
                        client.send_action_typing(chat_id);
                        client.send_text(chat_id, chunk.as_str());
                        tokio::time::sleep(Duration::from_millis(400)).await;
                    }
                    remember_in_history(tg_history, Message::assistant(reply.to_string()));
                    record_my_send(tg_history, tg_history_full, my_id, cfg, &[reply.to_string()]);
                } else {
                    warn!("[tg] фолбэк-текст НЕ отправлен: голосовой ввод требует голос");
                }
                break;
            }
            continue;
        }

        // Если дошли сюда — ни один tool не отправил финал и не обработан.
        // (обычно не случается при tool_choice=required + send_telegram_message)
        if is_idle {
            // Idle: canned-реплика тут была бы «спамом по часам» — молчим.
            warn!("[tg] idle: ни один tool не завершил ход — промолчали");
            break;
        }
        warn!("[tg] ни один tool не завершил ответ — fallback");
        let reply =
            "Сашенька, я тут немного зависла в мыслях... но я с тобой! 🥰 Расскажи ещё что-нибудь?";
        // Голосовой ввод → фолбэк-текст НЕ отправляем.
        if voice_text.is_none() {
            client.send_text(chat_id, reply);
            remember_in_history(tg_history, Message::assistant(reply.to_string()));
            record_my_send(tg_history, tg_history_full, my_id, cfg, &[reply.to_string()]);
        } else {
            warn!("[tg] фолбэк-текст НЕ отправлен: голосовой ввод требует голос");
        }
        return;
    }
}

/// Режим Telegram: авторизация, цикл update-ов, idle/sleep автономность.
#[allow(clippy::collapsible_if)]
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

    info!("[tg] слушаю update-ы (Ctrl-C для выхода)...");
    let tg_system_prompt = prompts.build_system_prompt();
    let mut tg_history: Vec<Message> = vec![Message::system(tg_system_prompt.clone())];
    let mut last_user_activity = Instant::now();
    // Активность ЛЮБАЯ (его сообщение ИЛИ наше собственное) — по ней
    // считаем «совсем тихо» и решаем, когда уснуть.
    let mut last_activity = Instant::now();
    let mut consolidated = false;
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
                            match consolidate_memory(
                                llm,
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
                            match furinya_diary::build_profile_memory(
                                llm,
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
                            let chat_turns: Vec<furinya_diary::ChatTurn> =
                                furinya_diary::ChatTurn::from_tg_full(
                                    &tg_history_full,
                                    &cfg.general.character_name,
                                    &cfg.general.papik_name,
                                );
                            match consolidate_memory(
                                llm,
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
                        let stim = serde_json::json!({
                            "@type": "updateNewMessage",
                            "message": {
                                "chat_id": papik,
                                "id": 0,
                                "sender_id": { "user_id": 0 },
                                "content": { "text": { "text": format!("[idle] [внутреннее] Сейчас {}:{} по времени Саши. Тишина: он не писал уже {} минут. Это НЕ сигнал «надо что-то написать» — это момент, когда ты МОЖЕШЬ написать первая, если сама хочется и есть что сказать. Если он сейчас спит или ему не до тебя — не пиши вовсе (stay_quiet). Если нечего сказать или не хочется — тоже промолчи (это нормально, никто не обидится). Не пиши «просто чтобы напомнить о себе», не спрашивай «ты тут?» и не жалуйся на ожидание. Написать можно ОДНУ короткую живую реплику: про свои дела, настроение, что-то милое про него — только если это правда твоё желание.{}", local_now.hour(), local_now.minute(), elapsed_any / 60, own_hint) } }
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
            let out_sender = v["message"]["sender_id"]["user_id"].as_i64().unwrap_or(0);
            if out_sender != my_id {
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
                // Порог: diary_token_count_trigger (токенов) × 4 символа.
                let overflow_limit =
                    cfg.misc.diary_token_count_trigger as usize * 4;
                let total_chars: usize = tg_history_full
                    .iter()
                    .map(|(_, _, t, _)| t.chars().count())
                    .sum();
                if !consolidated && total_chars >= overflow_limit {
                    info!("[tg] overflow: {} симв >= {} — консолидация", total_chars, overflow_limit);
                    consolidated = true;
                    let chat_turns: Vec<furinya_diary::ChatTurn> =
                        furinya_diary::ChatTurn::from_tg_full(
                            &tg_history_full,
                            &cfg.general.character_name,
                            &cfg.general.papik_name,
                        );
                    match consolidate_memory(
                        llm,
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
                        }
                        Err(e) => error!("[tg] overflow-консолидация ошибка: {}", e),
                    }
                    // Подрезка: хвост остаётся, старьё уходит в дневник.
                    const KEEP_LAST: usize = 40;
                    if tg_history_full.len() > KEEP_LAST {
                        let drop = tg_history_full.len() - KEEP_LAST;
                        tg_history_full.drain(..drop);
                    }
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

