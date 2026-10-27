//! Конфигурация ядра Furinya (миграция с Kuni C++).
//!
//! Формат TOML сохранён для обратной совместимости с промптами и настройками:
//! секции [general], [general.llm], [general.embedding], [misc], [capabilities].

use serde::Deserialize;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("не удалось прочитать файл конфига: {0}")]
    Io(#[from] std::io::Error),
    #[error("ошибка парсинга TOML: {0}")]
    Toml(#[from] toml::de::Error),
}

/// Эндпоинт OpenAI-совместимого API (LLM / embedding / vision и т.д.).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct EndpointConfig {
    #[serde(rename = "baseUrl", default)]
    pub base_url: String,
    #[serde(rename = "bearerKey", default)]
    pub bearer_key: String,
    #[serde(default)]
    pub model: String,
    /// Для SD: имя checkpoint (например animagineXL_v31.safetensors),
    /// передаётся через override_settings без глобального переключения модели.
    #[serde(rename = "checkpoint", default)]
    pub checkpoint: String,
    /// Резервная (fallback) модель: используется, если основная отдал HTTP-ошибку
    /// (429 rate-limit, 5xx и т.п.) или не ответила. См. LlmClient (furinya-llm).
    #[serde(rename = "fallbackModel", default)]
    pub fallback_model: String,
}

/// Секция [general]
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GeneralConfig {
    #[serde(default = "default_character_name")]
    pub character_name: String,
    #[serde(default = "default_character_nickname")]
    pub character_nickname: String,
    #[serde(default = "default_papik_name")]
    pub papik_name: String,
    /// Telegram username владельца (без `@`). Необязательный, но полезный
    /// второй признак: основной — `papik_chat_id`. Сравнение регистронезависимое,
    /// чтобы не привязывать код к конкретному имени/аккаунту.
    #[serde(default)]
    pub papik_username: String,
    #[serde(default)]
    pub papik_chat_id: i64,
    /// Сколько последних сообщений чата попадает в контекст модели напрямую.
    /// Остальная история (старше этого лимита) не шлётся в промпт — модель
    /// достаёт её через RAG (diary / per-чат выжимка). Экономит токены.
    #[serde(default = "default_tg_recent_limit")]
    pub tg_recent_limit: usize,
    /// Сколько сообщений истории чата грузить при старте (для per-чат выжимки и RAG).
    /// 0 = только последние 50 (один запрос). Больше = пагинация по 100/страница.
    #[serde(default = "default_tg_history_load_limit")]
    pub tg_history_load_limit: usize,
    /// Собственный ID аккаунта Furinya (user_id из TDLib). Используется чтобы
    /// не отвечать самой себе. Можно узнать из лога: "Logged in as user X".
    #[serde(default)]
    pub telegram_my_id: i64,
    #[serde(default)]
    pub telegram_api_id: i32,
    #[serde(default)]
    pub telegram_api_hash: String,
    #[serde(default = "default_lockdown")]
    pub lockdown: String,
    #[serde(default = "default_true")]
    pub telegram_enabled: bool,
    pub llm: LlmEndpoint,
    #[serde(default)]
    pub embedding: LlmEndpoint,
}

impl GeneralConfig {
    /// Telegram username владельца без `@`, если он задан в конфиге.
    pub fn papik_username_clean(&self) -> Option<&str> {
        let username = self.papik_username.trim().trim_start_matches('@');
        if username.is_empty() {
            None
        } else {
            Some(username)
        }
    }

    /// Отображаемое имя владельца: `papik_name`, иначе его Telegram-username,
    /// иначе нейтральное «собеседник». Ни одно конкретное имя в коде не зашито.
    pub fn papik_display_name(&self) -> String {
        let name = self.papik_name.trim();
        if !name.is_empty() {
            return name.to_string();
        }
        if let Some(username) = self.papik_username_clean() {
            return format!("@{username}");
        }
        "собеседник".to_string()
    }
}

/// LLM-эндпоинт в формате Kuni: [general.llm] -> endpoint.{baseUrl, bearerKey, model}
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LlmEndpoint {
    #[serde(default)]
    pub endpoint: EndpointConfig,
}

fn default_character_name() -> String {
    "Furinya".into()
}
fn default_character_nickname() -> String {
    "@furinya_chan".into()
}
fn default_papik_name() -> String {
    // Пусто = «не задано»: имя собеседника не хардкодится, а берётся из
    // papik_username / live-профиля Telegram (см. papik_display_name).
    String::new()
}
fn default_lockdown() -> String {
    "papik_only".into()
}
fn default_true() -> bool {
    true
}
fn default_tg_recent_limit() -> usize {
    10
}
fn default_tg_history_load_limit() -> usize {
    500
}

/// Секция [misc]
#[derive(Debug, Clone, Default, Deserialize)]
pub struct MiscConfig {
    #[serde(default = "default_true")]
    pub wake_up_on_pinned_chat: bool,
    #[serde(default = "default_true")]
    pub randomly_go_sleep: bool,
    #[serde(default = "default_tool_reminder_prob")]
    pub tool_reminder_probability: f32,
    #[serde(default = "default_diray_trigger")]
    pub diary_token_count_trigger: u32,
    /// Каталог с записями дневника (data/diary/*.md). Абсолютный или относительный к cwd.
    #[serde(default = "default_diary_dir")]
    pub diary_dir: String,
    #[serde(default)]
    pub diary_injection_max_length: u32,
    #[serde(default = "default_plagiarism")]
    pub diary_plagiarism_threshold: f32,
    #[serde(default = "default_relatedness")]
    pub diary_min_relatedness: f32,
    /// Сколько последних (по таймстампу) записей дневника держать на диске.
    /// Более старые УДАЛЯЮТСЯ при prune (но защищённые — активный good-контекст
    /// топ-K и corrupt-записи — не трогаются никогда). 0 = не удалять ничего.
    #[serde(default = "default_diary_keep_last")]
    pub diary_keep_last: usize,
    /// Сколько самых свежих НЕ-битых записей дневника шлётся в активный
    /// контекст модели (склеенных). Остальная память достаётся через RAG (#ask).
    #[serde(default = "default_active_top_k")]
    pub active_top_k: usize,
    #[serde(default = "default_chat_hist")]
    pub chat_max_history_length: u32,
    /// Диапазон имитации скорости печати (слов/мин) для задержки между
    /// дробными сообщениями в Telegram.
    #[serde(default = "default_typing_min")]
    pub typing_simulation_min_wpm: u32,
    #[serde(default = "default_typing_max")]
    pub typing_simulation_max_wpm: u32,
    /// Вероятность «живой опечатки»: сообщение уходит с одной случайной
    /// опечаткой в слове ≥5 букв и тут же редактируется на верный текст
    /// (эффект «перепечатала»). 0 = выключено.
    #[serde(default = "default_typo_prob")]
    pub liveliness_typo_probability: f32,
    /// Мягкий лимит токенов контекста (грубая оценка chars/4). При превышении —
    /// включается рефлексия/консолидация памяти и укорачивание истории.
    #[serde(default = "default_ctx_limit")]
    pub context_token_limit: u32,
    /// Максимальное число частей, на которые модель дробит ответ (каждое —
    /// отдельное сообщение в Telegram). Лишнее отбрасывается.
    #[serde(default = "default_max_parts")]
    pub max_message_parts: usize,
    /// Анти-повтор (порт kuni, send_telegram_message.cpp): эмбеддинг-проверка
    /// черновика ответа против её недавних реплик. Максимум похожести —
    /// anti_repeat_trigger_max, средняя — anti_repeat_trigger_avg. 1.0 = выкл.
    #[serde(default = "default_anti_repeat_max")]
    pub anti_repeat_trigger_max: f32,
    #[serde(default = "default_anti_repeat_avg")]
    pub anti_repeat_trigger_avg: f32,
    /// Сколько последних своих реплик держать в кэше эмбеддингов анти-повтора.
    #[serde(default = "default_anti_repeat_history")]
    pub anti_repeat_max_history: usize,
    /// Новостные каналы (usernames без @), которые бот читает для «актуальных
    /// разговоров»: Россия + Нижний Новгород. Пусто = новости выключены.
    #[serde(default)]
    pub news_channels: Vec<String>,
    /// Автоподписка на news_channels при старте (joinChat). false = только
    /// читать уже подписанные.
    #[serde(default = "default_true")]
    pub news_auto_join: bool,
    /// Период опроса каналов (сек). 0 = поллер выключен.
    #[serde(default = "default_news_poll")]
    pub news_poll_secs: u64,
    #[serde(default = "default_temp")]
    pub llm_temperature: f32,
    #[serde(default = "default_none_str")]
    pub llm_top_p: String,
    #[serde(default = "default_none_str")]
    pub llm_top_k: String,
    #[serde(default = "default_none_str")]
    pub llm_min_p: String,
    #[serde(default = "default_none_str")]
    pub presence_penalty: String,
    #[serde(default = "default_none_str")]
    pub repetition_penalty: String,
    #[serde(default = "default_timeout")]
    pub request_timeout_secs: String,
    /// Через сколько секунд молчания пользователя Furinya САМА пишет в чат
    /// (инициатива). 0 = никогда не писать сама.
    #[serde(default = "default_idle_msg")]
    pub idle_message_timeout_secs: u64,
    /// Через сколько секунд молчания включается режим сна (consolidate дневник
    /// + сообщение "иду спать"). 0 = никогда не спать.
    #[serde(default = "default_sleep_to")]
    pub sleep_timeout_secs: u64,
    /// Минимальная пауза (сек) перед тем, как ей будет ПРЕДЛОЖЕНО написать
    /// первой. Таймер только даёт шанс — пишет она или молчит, решает модель.
    #[serde(default = "default_idle_window_min")]
    pub idle_window_min_secs: u64,
    /// Максимальная пауза (сек) до того же шанса. Разброс между min и max —
    /// чтобы «писать первой» не выглядело как по часам.
    #[serde(default = "default_idle_window_max")]
    pub idle_window_max_secs: u64,
    /// Сколько своих инициативных сообщений подряд можно оставить без ответа
    /// (после этого — тишина, ждём его). Страховка от спама.
    #[serde(default = "default_idle_ping_max")]
    pub idle_ping_max: u32,
    /// Тихие часы: юзер обычно спит с этого часа (локальное время) до
    /// `quiet_hours_end` (не включается). В это время бот НЕ пишет первой,
    /// НЕ будит сообщениями и НЕ шлёт «спокойной ночи» — только тихо
    /// консолидирует память. 0 = выключено (или start == end).
    /// Окно через полночь: start=23, end=7.
    #[serde(default)]
    pub quiet_hours_start: u32,
    #[serde(default)]
    pub quiet_hours_end: u32,
    /// Ситуационная память: сколько последних заметок (<diary_dir>/situation/
    /// <chat_id>.md) подмешивать в контекст на первом ходу — непрерывность
    /// «что происходит между нами прямо сейчас». 0 = не подмешивать.
    #[serde(default = "default_situation_recent_n")]
    pub situation_recent_n: usize,
    /// Ситуационная память: сколько заметок достать семантически (top-K) по
    /// свежему сообщению — чтобы всплыло похожее по смыслу из прошлого, а не
    /// только хвост истории. 0 = выключить семантическую выборку.
    #[serde(default = "default_situation_top_k")]
    pub situation_top_k: usize,
    /// Ситуационная память: сколько последних заметок держать на диске
    /// (md + индекс). 0 = не удалять ничего (история растёт).
    #[serde(default)]
    pub situation_keep_entries: usize,
}

fn default_tool_reminder_prob() -> f32 {
    0.02
}
fn default_diray_trigger() -> u32 {
    6000
}
fn default_diary_dir() -> String {
    "data/diary".to_string()
}
fn default_plagiarism() -> f32 {
    0.97
}
fn default_relatedness() -> f32 {
    // Центрированный косинус mistral-embed: фон ≈ 0, релевантный топ 0.2–0.35.
    0.15
}
fn default_diary_keep_last() -> usize {
    10
}
fn default_active_top_k() -> usize {
    5
}
fn default_chat_hist() -> u32 {
    2000
}
fn default_typing_min() -> u32 {
    120
}
fn default_typing_max() -> u32 {
    150
}
fn default_typo_prob() -> f32 {
    0.2
}
fn default_ctx_limit() -> u32 {
    20000
}
fn default_max_parts() -> usize {
    6
}

fn default_anti_repeat_max() -> f32 {
    0.95
}

fn default_anti_repeat_avg() -> f32 {
    0.85
}

fn default_anti_repeat_history() -> usize {
    32
}

fn default_news_poll() -> u64 {
    1800
}
fn default_temp() -> f32 {
    0.2
}
fn default_none_str() -> String {
    "none".into()
}
fn default_timeout() -> String {
    "00:15:00.000000".into()
}
fn default_idle_msg() -> u64 {
    300 // 5 минут молчания → Furinya сама пишет
}
fn default_sleep_to() -> u64 {
    1800 // 30 минут молчания → режим сна + consolidate
}
fn default_idle_window_min() -> u64 {
    300 // 5 минут — раньше шанса «написать первой» не будет
}
fn default_idle_window_max() -> u64 {
    900 // и не позже 15 минут (окно случайное → не «по часам»)
}
fn default_idle_ping_max() -> u32 {
    3 // максимум 3 своих сообщения без ответа — дальше тишина
}
fn default_situation_recent_n() -> usize {
    6 // последние 6 заметок — непрерывность ситуации
}
fn default_situation_top_k() -> usize {
    3 // + семантический top-3 по свежему сообщению
}

/// Секция [capabilities] (пока только заглушки, расширяем по ходу)
#[derive(Debug, Clone, Deserialize, Default)]
pub struct CapabilitiesConfig {
    #[serde(default)]
    pub web_search: WebSearchConfig,
    #[serde(default)]
    pub vision: VisionConfig,
    #[serde(default)]
    pub use_stickers: ToggleConfig,
    #[serde(default)]
    pub take_photo: TakePhotoConfig,
    #[serde(default)]
    pub hearing: HearingConfig,
    #[serde(default)]
    pub record_voice: RecordVoiceConfig,
    #[serde(default)]
    pub proxy: ToggleConfig,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ToggleConfig {
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WebSearchConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub ollama_bearer_key: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct VisionConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub llm_image_to_text: EndpointConfig,
    #[serde(default)]
    pub llm_image_to_text_cheap: EndpointConfig,
    /// Максимальная сторона фото (px) перед отправкой в vision-модель.
    /// БЕЗ хардкода магических чисел (как было в kuni: вшитые 64 → 1000+ токенов
    /// на любую фотку). TDLib и так даёт photo.sizes[-1] разумного размера,
    /// но здесь явный лимит на всякий (0 или None = не даунскейлить).
    #[serde(default)]
    pub max_dim: u32,
    /// Качество JPEG при перекодировке (1..100). Меньше = меньше токенов.
    #[serde(default = "default_jpeg_quality")]
    pub jpeg_quality: u8,
}

fn default_jpeg_quality() -> u8 {
    80
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TakePhotoConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Бэкенд генерации: "aihorde" (Stable Horde, бесплатный, NSFW) или
    /// "a1111" (локальный AUTOMATIC1111 / Forge WebUI).
    #[serde(default = "default_sd_backend")]
    pub backend: String,
    #[serde(default)]
    pub sd: EndpointConfig,
    #[serde(default)]
    pub aihorde: AiHordeConfig,
    /// Параметры рендера локального sd-server. Раньше были зашиты в код
    /// (steps=8, cfg=1.0, euler_a, 512x640, две LoRA) — менять их приходилось
    /// пересборкой. Теперь крутятся из конфига.
    #[serde(default)]
    pub sd_render: SdRenderConfig,
    /// Проверять готовое фото зрением и при явном браке перегенерировать
    /// с исправленным промптом. Так работал старый «плюсовой» проект: модель
    /// смотрит на свою картинку, критикует её и правит промпт следующего кадра.
    #[serde(default = "default_true")]
    pub review: bool,
    /// Сколько раз разрешено перегенерировать после критики (0 = без повтора).
    #[serde(default = "default_review_retries")]
    pub review_retries: u32,
}

fn default_review_retries() -> u32 {
    1
}

/// Параметры рендера локального sd-server (stable-diffusion.cpp).
#[derive(Debug, Clone, Deserialize)]
pub struct SdRenderConfig {    /// Шаги. Hyper-SD 8-step: steps=8 при cfg≈1.0. Обычный сэмплинг: 25-30 при cfg 6-8.
    #[serde(default = "default_sd_steps")]
    pub steps: u32,
    #[serde(default = "default_sd_cfg")]
    pub cfg_scale: f32,
    #[serde(default = "default_sd_sampler")]
    pub sampler_name: String,
    #[serde(default = "default_sd_width")]
    pub width: u32,
    #[serde(default = "default_sd_height")]
    pub height: u32,
    #[serde(default = "default_sd_clip_skip")]
    pub clip_skip: i32,
    /// LoRA: путь (имя файла в lora-model-dir) + сила.
    #[serde(default)]
    pub loras: Vec<SdLoraConfig>,
    /// Сид: -1 = случайный (нужен для разнообразия).
    #[serde(default = "default_sd_seed")]
    pub seed: i64,
}

/// Одна LoRA для локального sd-server.
///
/// `deny_unknown_fields` — страховка от TOML-ошибки раздела: dotted-ключи,
/// стоящие после `[[...loras]]`, попадают в последний элемент массива. Так
/// `record_voice.enabled = true` молча оказывался полем LoRA, а голос читался
/// выключенным. Теперь такой конфиг падает на старте, а не деградирует тихо.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdLoraConfig {
    /// Имя файла в lora-model-dir (например "furina.safetensors").
    pub path: String,
    #[serde(default = "default_lora_multiplier")]
    pub multiplier: f32,
}

fn default_sd_steps() -> u32 {
    8
}
fn default_sd_cfg() -> f32 {
    1.0
}
fn default_sd_sampler() -> String {
    "euler_a".to_string()
}
fn default_sd_width() -> u32 {
    512
}
fn default_sd_height() -> u32 {
    640
}
fn default_sd_clip_skip() -> i32 {
    2
}
fn default_sd_seed() -> i64 {
    -1
}
fn default_lora_multiplier() -> f32 {
    0.7
}

impl Default for SdRenderConfig {
    fn default() -> Self {
        Self {
            steps: default_sd_steps(),
            cfg_scale: default_sd_cfg(),
            sampler_name: default_sd_sampler(),
            width: default_sd_width(),
            height: default_sd_height(),
            clip_skip: default_sd_clip_skip(),
            loras: Vec::new(),
            seed: default_sd_seed(),
        }
    }
}

/// Конфиг AI Horde (https://aihorde.net/api/v2). Бесплатная краудсорс-генерация.
/// Анонимный ключ "0000000000" работает, но с минимальным приоритетом в очереди —
/// для приоритета зарегистрируй ключ на https://stablehorde.net/register.
#[derive(Debug, Clone, Deserialize)]
pub struct AiHordeConfig {
    /// API-ключ Horde (анонимный: "0000000000").
    #[serde(default = "default_aihorde_key")]
    pub apikey: String,
    /// Заголовок Client-Agent (обязателен): "имя:версия:контакт".
    #[serde(default = "default_aihorde_agent")]
    pub client_agent: String,
    /// Модели для генерации (список можно получить: GET /api/v2/models/all).
    /// Пустой список = любой подходящий воркер.
    #[serde(default)]
    pub models: Vec<String>,
    /// Разрешить NSFW-контент (воркеры с filter_sfw=false).
    #[serde(default = "default_true")]
    pub nsfw: bool,
    /// Цензурировать ли NSFW со стороны Horde (false = как есть).
    #[serde(default)]
    pub censor_nsfw: bool,
    /// Размеры/сэмплинг по умолчанию (может быть переопределён промптом модели).
    #[serde(default = "default_aihorde_width")]
    pub width: u32,
    #[serde(default = "default_aihorde_height")]
    pub height: u32,
    #[serde(default = "default_aihorde_steps")]
    pub steps: u32,
    #[serde(default = "default_aihorde_cfg")]
    pub cfg_scale: f32,
    #[serde(default = "default_aihorde_sampler")]
    pub sampler_name: String,
    /// Интервал опроса /generate/check (мс).
    #[serde(default = "default_aihorde_poll_ms")]
    pub poll_interval_ms: u64,
    /// Сколько ждать завершения генерации всего (сек). Анонимный ключ в
    /// очереди может стоять долго.
    #[serde(default = "default_aihorde_timeout")]
    pub poll_timeout_secs: u64,
}

impl Default for AiHordeConfig {
    fn default() -> Self {
        Self {
            apikey: default_aihorde_key(),
            client_agent: default_aihorde_agent(),
            models: Vec::new(),
            nsfw: true,
            censor_nsfw: false,
            width: default_aihorde_width(),
            height: default_aihorde_height(),
            steps: default_aihorde_steps(),
            cfg_scale: default_aihorde_cfg(),
            sampler_name: default_aihorde_sampler(),
            poll_interval_ms: default_aihorde_poll_ms(),
            poll_timeout_secs: default_aihorde_timeout(),
        }
    }
}

fn default_aihorde_key() -> String {
    "0000000000".into()
}
fn default_aihorde_agent() -> String {
    "furinya:0.1.0:anonymous".into()
}
fn default_aihorde_width() -> u32 {
    512
}
fn default_aihorde_height() -> u32 {
    704
}
fn default_aihorde_steps() -> u32 {
    25
}
fn default_aihorde_cfg() -> f32 {
    7.0
}
fn default_aihorde_sampler() -> String {
    "k_euler_a".into()
}
fn default_aihorde_poll_ms() -> u64 {
    3000
}
fn default_aihorde_timeout() -> u64 {
    900
}

fn default_sd_backend() -> String {
    "aihorde".into()
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct HearingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub llm_audio_to_text: EndpointConfig,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RecordVoiceConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Синтез речи: "fish" (Fish Audio, рус/клон голоса по reference_id)
    /// или "voxtral" (Mistral TTS — но РУССКОГО в списке языков нет!)
    /// или "openai". Пусто = синтез не используется.
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub elevenlabs: ElevenLabsConfig,
    #[serde(default)]
    pub openai: OpenAiTtsConfig,
    #[serde(default)]
    pub fish: FishTtsConfig,
    #[serde(default)]
    pub voxtral: VoxtralTtsConfig,
}

/// Fish Audio (https://fish.audio): POST https://api.fish.audio/v1/tts.
/// reference_id — UUID голосовой модели, созданной в Fish Audio дашборде
/// (или через клон с furina_ref.wav — см. docs/voice-reference-guide.md).
/// Бесплатный тариф: строка модели "s2.1-pro-free" (fair-use, 83 языка,
/// клонирование бесплатно). ВНИМАНИЕ: если model НЕ задан, Fish откатывается
/// на ПЛАТНЫЙ s2.1-pro — поэтому свободная строка описана в конфиге явно.
/// FishTtsConfig НЕ derive(Default): пустая модель = платный откат Fish,
/// поэтому Default обязан подставлять "s2.1-pro-free", а не "".
#[derive(Debug, Clone, Deserialize)]
pub struct FishTtsConfig {
    /// Bearer-ключ Fish Audio API.
    #[serde(default)]
    pub key: String,
    /// UUID голосовой модели (voice model) Furinya.
    #[serde(rename = "referenceId", default)]
    pub reference_id: String,
    /// Строка модели Fish Audio. "s2.1-pro-free" = бесплатный (fair-use).
    #[serde(default = "default_fish_model")]
    pub model: String,
    /// Битрейт получаемого MP3 (64/128/192...).
    #[serde(default = "default_fish_bitrate")]
    pub mp3_bitrate: u32,
    /// temperature сэмплинга TTS (0.7 — нейтрально).
    #[serde(default = "default_fish_temperature")]
    pub temperature: f32,
    /// top_p сэмплинга TTS (0.7 — нейтрально).
    #[serde(default = "default_fish_temperature")]
    pub top_p: f32,
}

impl Default for FishTtsConfig {
    fn default() -> Self {
        Self {
            key: String::new(),
            reference_id: String::new(),
            model: default_fish_model(),
            mp3_bitrate: default_fish_bitrate(),
            temperature: default_fish_temperature(),
            top_p: default_fish_temperature(),
        }
    }
}

fn default_fish_model() -> String {
    "s2.1-pro-free".into()
}

fn default_fish_bitrate() -> u32 {
    128
}
fn default_fish_temperature() -> f32 {
    0.7
}

/// Voxtral TTS (Mistral API, POST /v1/audio/speech).
/// ВНИМАНИЕ: официально поддерживает en, fr, de, es, nl, pt, it, hi, ar —
/// РУССКОГО НЕТ. Клон голоса с русским reference => деградация.
/// Оставлено как экспериментальный бэкенд: ref_audio (base64 per-request) или
/// сохранённый voice UUID.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct VoxtralTtsConfig {
    /// Путь к reference-аудио рядом с бинарником (furina_ref.wav).
    #[serde(rename = "refAudioPath", default)]
    pub ref_audio_path: String,
    /// UUID сохранённого голоса (создаётся через POST /v1/audio/voices).
    /// Если задан — ref_audio не нужен.
    #[serde(rename = "voiceId", default)]
    pub voice_id: String,
    /// Формат ответа ("mp3" | "wav" | "pcm").
    #[serde(rename = "responseFormat", default = "default_voxtral_format")]
    pub response_format: String,
}

fn default_voxtral_format() -> String {
    "mp3".into()
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ElevenLabsConfig {
    #[serde(default)]
    pub key: String,
    #[serde(default)]
    pub voice_id: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct OpenAiTtsConfig {
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub key: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub voice: String,
}

/// Корневая конфигурация
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub misc: MiscConfig,
    #[serde(default)]
    pub capabilities: CapabilitiesConfig,
}

impl Config {
    /// Загружает конфиг из файла по пути.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let cfg: Config = toml::from_str(&content)?;
        Ok(cfg)
    }

    /// Возвращает URL чата LLM из секции general.llm.
    pub fn llm_base_url(&self) -> &str {
        &self.general.llm.endpoint.base_url
    }

    pub fn llm_model(&self) -> &str {
        &self.general.llm.endpoint.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn misplaced_capability_keys_fail_loudly() {
        // Dotted-ключи после [[...loras]] TOML кладёт в последний элемент
        // массива. Без deny_unknown_fields такой конфиг молча выключал голос.
        let broken = r#"
[capabilities]
take_photo.enabled = true

[[capabilities.take_photo.sd_render.loras]]
path = "furina.safetensors"
multiplier = 0.75

record_voice.enabled = true
"#;
        let err = toml::from_str::<Config>(broken).expect_err("лишний ключ должен падать");
        assert!(
            err.to_string().contains("record_voice"),
            "ошибка должна называть лишний ключ, а не молча его глотать: {err}"
        );
    }

    #[test]
    fn parse_existing_config() {
        // Читаем реальный config.toml из корня проекта относительно CWD теста.
        let path = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let root = std::path::Path::new(&path)
            .ancestors()
            .nth(2)
            .unwrap()
            .join("config.toml");
        let cfg = Config::load(&root).expect("config.toml должен парситься");
        assert_eq!(cfg.general.character_name, "Furinya");
        assert_eq!(cfg.general.papik_chat_id, 5005580127);
        // Основная LLM — Mistral API: ministral-14b (937.5k TPM / 30 rpm, качество),
        // резерв — ministral-8b (625k TPM / 188 rpm, если 14b ловит 429/5xx).
        assert!(cfg.general.llm.endpoint.base_url.contains("api.mistral.ai"));
        assert_eq!(cfg.general.llm.endpoint.model, "ministral-14b-2512");
        assert_eq!(cfg.general.llm.endpoint.fallback_model, "ministral-8b-2512");
        // Эмбеддинги: Mistral (mistral-embed, 1024) — тот же ключ, что у LLM.
        // Gemini-квота кончилась (429), локальный llama.cpp на этом хосте
        // считал запись ~4.6 с и грел железо. Модель обязана лежать ВНУТРИ
        // endpoint.* — иначе в запрос уходит пустой model и бэкенд отвечает 400.
        assert_eq!(cfg.general.embedding.endpoint.model, "mistral-embed");
        assert!(cfg
            .general
            .embedding
            .endpoint
            .base_url
            .contains("api.mistral.ai"));
        // Vision на Mistral API: ПРОВЕРЕНО живьём — ministral-14b/8b
        // мультимодальные (писали описания картинок), medium/small — 429.
        assert_eq!(
            cfg.capabilities
                .vision
                .llm_image_to_text
                .model,
            "ministral-14b-2512"
        );
        assert!(cfg.capabilities.take_photo.enabled);
        // Основной бэкенд фото — локальный sd-server (SD1.5, 512×640,
        // meinahentai): облако (Horde) больше не используется.
        assert_eq!(cfg.capabilities.take_photo.backend, "sd");
        assert!(cfg
            .capabilities
            .take_photo
            .sd
            .base_url
            .contains("192.168.1.172:8082"));
        // Параметры рендера — из конфига (раньше были зашиты в код).
        assert_eq!(cfg.capabilities.take_photo.sd_render.steps, 8);
        assert_eq!(cfg.capabilities.take_photo.sd_render.clip_skip, 2);
        assert!(!cfg.capabilities.take_photo.sd_render.loras.is_empty());
    }
}
