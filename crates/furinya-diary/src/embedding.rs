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
        Self {
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            bearer_key: bearer,
            model: cfg.model.clone(),
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
}
