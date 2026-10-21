//! Клиент для генерации эмбеддингов (векторной памяти RAG).
//!
//! Использует OpenAI-совместимый эндпоинт `/v1/embeddings`
//! (в kuni — отдельный легковесный сервер bge-m3 на :8091).

use anyhow::Context;
use serde::Deserialize;
use std::time::Duration;

use furinya_core::EndpointConfig;

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
    client: reqwest::Client,
}

impl EmbeddingClient {
    pub fn from_config(cfg: &EndpointConfig) -> Self {
        // Env-оверрайд: bearerKey пустой в config → берём MISTRAL_API_KEY
        // (ключи хранятся только в env, spec §1).
        let bearer = if cfg.bearer_key.trim().is_empty() {
            std::env::var("MISTRAL_API_KEY").unwrap_or_default()
        } else {
            cfg.bearer_key.clone()
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
        Self {
            base_url,
            bearer_key: bearer,
            model,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("не удалось создать HTTP-клиент для embeddings"),
        }
    }

    /// Возвращает эмбеддинг одного текста.
    pub async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let url = format!("{}/embeddings", self.base_url);
        let mut req = self.client.post(&url).json(&serde_json::json!({
            "model": self.model,
            "input": text,
        }));
        if !self.bearer_key.is_empty() {
            req = req.bearer_auth(&self.bearer_key);
        }
        let resp = req
            .send()
            .await
            .with_context(|| format!("запрос эмбеддинга к {}", url))?;
        if !resp.status().is_success() {
            anyhow::bail!(
                "эмбеддинг API вернул {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            );
        }
        let parsed: EmbeddingResponse = resp.json().await.context("парсинг ответа эмбеддинга")?;
        parsed
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| anyhow::anyhow!("пустой ответ эмбеддинга"))
    }

    /// Эмбеддинги для пачки текстов одним запросом OpenAI-совместимый:
    /// `input: [ ... ]`). Mistral возвращает data[] в порядке входных строк.
    pub async fn embed_batch(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/embeddings", self.base_url);
        let mut req = self.client.post(&url).json(&serde_json::json!({
            "model": self.model,
            "input": texts,
        }));
        if !self.bearer_key.is_empty() {
            req = req.bearer_auth(&self.bearer_key);
        }
        let resp = req
            .send()
            .await
            .with_context(|| format!("запрос пачки эмбеддингов к {}", url))?;
        if !resp.status().is_success() {
            anyhow::bail!(
                "эмбеддинг API вернул {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            );
        }
        let parsed: EmbeddingResponse = resp.json().await.context("парсинг пачки эмбеддингов")?;
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
