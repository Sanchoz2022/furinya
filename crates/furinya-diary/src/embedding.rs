//! Клиент для генерации эмбеддингов (векторной памяти RAG).
//!
//! Использует OpenAI-совместимый эндпоинт `/v1/embeddings` (сейчас —
//! Mistral `mistral-embed`, 1024 измерения; тот же ключ, что у LLM).
//!
//! У провайдера жёсткий лимит ~1 запрос/с (~60/мин), поэтому все вызовы идут
//! через общий на процесс шлюз (см. `rate_gate`): пачки считаются одним
//! запросом, а параллельные компоненты не ловят 429 одновременно.
//!
//! Mistral embeddings периодически отвечают `429`/`5xx`. Без защиты это молча
//! выключало всю векторную память (RAG, анти-повтор, консолидация дневника):
//! каждая проверка писала «пропускаю», модель оставалась без нового материала
//! и начинала повторяться. Два уровня защиты:
//!   1. Ретрай с бэкоффом на 429/5xx — короткая перегрузка переживается;
//!   2. Размыкатель (circuit breaker): если сервис лежит, падаем БЫСТРО и не
//!      тормозим ход на десятки секунд; через cooldown пробуем снова.

use anyhow::Context;
use serde::Deserialize;
use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};
use std::time::Duration;

use furinya_core::EndpointConfig;

/// Сколько раз пробуем один запрос эмбеддинга, прежде чем сдаться.
/// Две попытки (одна пауза 1с) — компромисс: короткий транзиентный 429
/// переживаем, но не висим секунды, когда сервис лежит.
const EMBED_MAX_ATTEMPTS: u32 = 2;
/// После стольких подряд неудачных вызовов размыкаем цепь. Эмбеддинги —
/// один общий сервис, так что двух сбоев достаточно, чтобы не долбить его
/// каждым компонентом (анти-повтор, RAG, консолидация) на каждом ходу.
const BREAKER_FAIL_THRESHOLD: u32 = 2;
/// Сколько секунд не трогаем сервис, когда цепь разомкнута.
const BREAKER_COOLDOWN_SECS: i64 = 60;

/// Unix-время, до которого эмбеддинги считаются недоступными (0 — доступны).
static BREAKER_UNTIL: AtomicI64 = AtomicI64::new(0);
/// Счётчик подряд идущих неудач.
static BREAKER_FAILS: AtomicU32 = AtomicU32::new(0);

/// Минимальный интервал между HTTP-запросами к `/embeddings`.
///
/// Mistral на этом ключе даёт ~60 запросов/мин, то есть ~1 запрос/с. Пачка
/// (`input: [...]`) считается ОДНИМ запросом, поэтому переиндексация и
/// анти-повтор укладываются в лимит, если не долбить сервис одиночными
/// вызовами. Шлюз общий на процесс: параллельные компоненты (RAG, анти-повтор,
/// ситуационная память) встают в очередь, а не ловят 429 одновременно.
const EMBED_MIN_INTERVAL: Duration = Duration::from_millis(1100);

/// Время, раньше которого следующий запрос уходить не должен.
static EMBED_NEXT_AT: tokio::sync::Mutex<Option<tokio::time::Instant>> =
    tokio::sync::Mutex::const_new(None);

/// Пропускает запрос не чаще одного раза в `EMBED_MIN_INTERVAL`.
///
/// Слот резервируется под мьютексом и сразу отпускается — ожидание идёт уже
/// без блокировки, поэтому медленный фон не выстраивает всех в мёртвую очередь.
async fn rate_gate() {
    let slot = {
        let mut next = EMBED_NEXT_AT.lock().await;
        let now = tokio::time::Instant::now();
        let slot = match *next {
            Some(t) if t > now => t,
            _ => now,
        };
        *next = Some(slot + EMBED_MIN_INTERVAL);
        slot
    };
    let now = tokio::time::Instant::now();
    if slot > now {
        tokio::time::sleep(slot - now).await;
    }
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Сервис недавно падал — не ждём таймаутов зря.
fn breaker_open() -> bool {
    now_unix() < BREAKER_UNTIL.load(Ordering::Relaxed)
}

fn breaker_on_success() {
    BREAKER_FAILS.store(0, Ordering::Relaxed);
    BREAKER_UNTIL.store(0, Ordering::Relaxed);
}

fn breaker_on_failure() {
    let fails = BREAKER_FAILS.fetch_add(1, Ordering::Relaxed) + 1;
    if fails >= BREAKER_FAIL_THRESHOLD {
        BREAKER_UNTIL.store(now_unix() + BREAKER_COOLDOWN_SECS, Ordering::Relaxed);
        tracing::warn!(
            "[embed] эмбеддинги недоступны ({} подряд) — пауза {}с",
            fails,
            BREAKER_COOLDOWN_SECS
        );
    }
}

/// Пауза перед повторной попыткой: 1с (дальше 2с, 4с — если поднять число
/// попыток).
fn retry_backoff(attempt: u32) -> Duration {
    Duration::from_secs(1u64 << (attempt.saturating_sub(1).min(4)))
}

/// Сколько ждать после 429. Провайдер присылает точное время в `Retry-After`
/// (Mistral) или в тексте («Please retry in 46.8s», Google). Это время почти
/// всегда БОЛЬШЕ нашего бэкоффа: при лимите ~60 запросов/мин повтор через
/// секунду гарантированно снова получит 429. `patient` (фоновая
/// переиндексация) ждёт сколько скажут, обычный ход — не дольше 8с, чтобы не
/// подвешивать ответ.
fn retry_delay(header_secs: Option<u64>, body: &str, patient: bool) -> Duration {
    let parsed = header_secs
        .map(|s| s as f64)
        .or_else(|| parse_retry_in(body));
    let cap = if patient { 180.0 } else { 8.0 };
    Duration::from_secs_f64(parsed.unwrap_or(5.0).clamp(1.0, cap))
}

/// Достаёт секунды из «Please retry in 46.811422194s».
fn parse_retry_in(body: &str) -> Option<f64> {
    let key = "Please retry in ";
    let i = body.find(key)? + key.len();
    let rest = &body[i..];
    let end = rest.find('s')?;
    rest[..end].trim().parse().ok()
}

/// Прокси для внешних эмбеддингов. Google напрямую из РФ недоступен, поэтому
/// запросы к нему идут через Mihomo. Порядок: FURINYA_EMBED_PROXY →
/// FURINYA_WEB_PROXY → HTTPS_PROXY/ALL_PROXY. Локальные адреса (llama.cpp на
/// 127.0.0.1, sd-api в LAN) прокси не используют — иначе сломается локальный
/// эмбеддер.
fn proxy_for(base_url: &str) -> Option<String> {
    let local = base_url.contains("127.0.0.1")
        || base_url.contains("localhost")
        || base_url.contains("192.168.")
        || base_url.contains("[::1]");
    if local {
        return None;
    }
    [
        "FURINYA_EMBED_PROXY",
        "FURINYA_WEB_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ]
    .iter()
    .find_map(|k| {
        std::env::var(k)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
    })
}

/// Ответ OpenAI-совместимого API эмбеддингов.
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Клиент для получения векторных представлений текста.
pub struct EmbeddingClient {
    base_url: String,
    bearer_key: String,
    model: String,
    /// Фоновая переиндексация: на 429 ждём столько, сколько просит провайдер,
    /// и делаем больше попыток (миграция памяти не должна падать из-за
    /// минутного лимита токенов). Для хода бота — false.
    patient_retry: bool,
    client: reqwest::Client,
}

impl EmbeddingClient {
    pub fn from_config(cfg: &EndpointConfig) -> Self {
        // Env-оверрайд: bearerKey пустой в config → берём ключ из env
        // (ключи хранятся только в env, spec §1). Для Google — GEMINI_API_KEY,
        // для Mistral — MISTRAL_API_KEY. Локальному серверу (llama.cpp) ключ
        // не нужен, и слать ему чужой Bearer незачем.
        let bearer = if !cfg.bearer_key.trim().is_empty() {
            cfg.bearer_key.clone()
        } else if cfg.base_url.contains("generativelanguage") {
            let k = std::env::var("GEMINI_API_KEY").unwrap_or_default();
            if k.trim().is_empty() {
                tracing::warn!(
                    "[embed] GEMINI_API_KEY не задан, а endpoint.baseUrl указывает на Google — \
                     эмбеддинги будут получать 401 (память/RAG не работают)"
                );
            }
            k
        } else if cfg.base_url.trim().is_empty() || cfg.base_url.contains("mistral.ai") {
            std::env::var("MISTRAL_API_KEY").unwrap_or_default()
        } else {
            String::new()
        };
        // Страховка от тихой поломки: без модели Mistral отвечает
        // 400 «Invalid model: None», и консолидация откатывается в ноль.
        // Пустая модель → дефолт (или env MISTRAL_EMBED_MODEL).
        let model = if cfg.model.trim().is_empty() {
            let m = std::env::var("MISTRAL_EMBED_MODEL")
                .unwrap_or_else(|_| "mistral-embed-2312".to_string());
            tracing::warn!(
                "[embed] модель эмбеддингов не задана в конфиге (endpoint.model) — беру {}",
                m
            );
            m
        } else {
            cfg.model.clone()
        };
        let base_url = if cfg.base_url.trim().is_empty() {
            tracing::warn!("[embed] endpoint.baseUrl не задан — беру https://api.mistral.ai/v1/");
            "https://api.mistral.ai/v1".to_string()
        } else {
            cfg.base_url.trim_end_matches('/').to_string()
        };
        let mut builder = reqwest::Client::builder().timeout(Duration::from_secs(60));
        if let Some(proxy) = proxy_for(&base_url) {
            match reqwest::Proxy::all(&proxy) {
                Ok(px) => {
                    builder = builder.proxy(px);
                    tracing::info!("[embed] внешний эмбеддер через прокси (из env)");
                }
                Err(e) => tracing::warn!("[embed] не удалось разобрать прокси из env: {e}"),
            }
        }
        Self {
            base_url,
            bearer_key: bearer,
            model,
            patient_retry: false,
            client: builder
                .build()
                .expect("не удалось создать HTTP-клиент для embeddings"),
        }
    }

    /// Терпеливая копия клиента для фоновой переиндексации: при 429 ждёт
    /// `Retry-After` провайдера (до 180с) и делает до 6 попыток вместо 2.
    /// `reqwest::Client` внутри — Arc, клонирование дешёвое.
    pub fn patient(&self) -> Self {
        Self {
            base_url: self.base_url.clone(),
            bearer_key: self.bearer_key.clone(),
            model: self.model.clone(),
            patient_retry: true,
            client: self.client.clone(),
        }
    }

    /// Идентификатор модели эмбеддингов (`baseUrl|model`) — для миграции
    /// векторной памяти при смене эмбеддера. Дневник хранит его в
    /// `data/diary/.embed_model` и сверяет при старте.
    pub fn identity(&self) -> String {
        format!("{}|{}", self.base_url, self.model)
    }

    /// Один POST `/embeddings` с ретраем и размыкателем.
    ///
    /// Размыкатель общий на процесс (static): когда Mistral лежит, все
    /// компоненты (RAG, анти-повтор, консолидация) падают быстро, а не висят
    /// секундами на каждом ходу.
    async fn post(
        &self,
        input: serde_json::Value,
        timeout: Duration,
    ) -> anyhow::Result<EmbeddingResponse> {
        if breaker_open() {
            anyhow::bail!("эмбеддинги временно недоступны (размыкатель: недавний 429/5xx)");
        }
        let url = format!("{}/embeddings", self.base_url);
        let attempts = if self.patient_retry { 6 } else { EMBED_MAX_ATTEMPTS };
        let mut last_err: anyhow::Error = anyhow::anyhow!("эмбеддинг: попытки не выполнены");
        for attempt in 1..=attempts {
            // Общий лимит провайдера: ~1 запрос/с. Ретраи тоже проходят шлюз.
            rate_gate().await;
            let mut req = self
                .client
                .post(&url)
                .timeout(timeout)
                .json(&serde_json::json!({
                    "model": self.model,
                    "input": input,
                }));
            if !self.bearer_key.is_empty() {
                req = req.bearer_auth(&self.bearer_key);
            }
            match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let parsed: EmbeddingResponse =
                            resp.json().await.context("парсинг ответа эмбеддинга")?;
                        breaker_on_success();
                        return Ok(parsed);
                    }
                    let retry_after = resp
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|s| s.trim().parse::<u64>().ok());
                    let body = resp.text().await.unwrap_or_default();
                    last_err = anyhow::anyhow!("эмбеддинг API вернул {status}: {body}");
                    let retryable = status.as_u16() == 429 || status.is_server_error();
                    if retryable && attempt < attempts {
                        let wait = if status.as_u16() == 429 {
                            retry_delay(retry_after, &body, self.patient_retry)
                        } else {
                            retry_backoff(attempt)
                        };
                        tracing::warn!(
                            "[embed] {status} — жду {wait:?} и ретраю ({attempt}/{attempts})"
                        );
                        tokio::time::sleep(wait).await;
                        continue;
                    }
                }
                Err(e) => {
                    last_err =
                        anyhow::Error::new(e).context(format!("запрос эмбеддинга к {url}"));
                    if attempt < attempts {
                        let wait = retry_backoff(attempt);
                        tracing::warn!(
                            "[embed] сетевая ошибка — жду {wait:?} и ретраю ({attempt}/{attempts})"
                        );
                        tokio::time::sleep(wait).await;
                        continue;
                    }
                }
            }
            break;
        }
        breaker_on_failure();
        Err(last_err)
    }

    /// Возвращает эмбеддинг одного текста.
    ///
    /// Таймаут 90с: одиночный текст — это запрос RAG / анти-повтора / одной
    /// записи при консолидации (короткие), но защищаем ход от зависания.
    pub async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let parsed = self
            .post(serde_json::json!(text), Duration::from_secs(90))
            .await?;
        parsed
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| anyhow::anyhow!("пустой ответ эмбеддинга"))
    }

    /// Эмбеддинги для пачки текстов одним запросом OpenAI-совместимым:
    /// `input: [ ... ]`). Mistral возвращает data[] в порядке входных строк.
    ///
    /// Таймаут 300с: сервер обрабатывает элементы пачки ПОСЛЕДОВАТЕЛЬНО, и
    /// длинные записи (пересчёт/миграция) на слабом CPU идут десятками секунд.
    pub async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let parsed = self
            .post(serde_json::json!(texts), Duration::from_secs(300))
            .await?;
        let out: Vec<Vec<f32>> = parsed.data.into_iter().map(|d| d.embedding).collect();
        if out.len() != texts.len() {
            anyhow::bail!(
                "пачка эмбеддингов: вернулось {} векторов из {} текстов",
                out.len(),
                texts.len()
            );
        }
        Ok(out)
    }
}