//! Авторизация в TDLib через JSON FFI.
//!
//! Протокол (упрощённо):
//! - Инициализация (setTdlibParameters + setDatabaseEncryptionKey) уже сделана
//!   в `TdClient::start()` через `execute`.
//! - Ждём updateAuthorizationState:
//!   - waitPhoneNumber / waitCode / waitPassword — запрашиваем у пользователя
//!   - ready — авторизация завершена
//!
//! В терминале запросы ввода идут через stdin; в проде (Telegram-бот)
//! код/пароль придут извне (от владельца).

use crate::{TdClient, build_td_params};
use serde_json::Value;
use std::io::{self, Write};
use std::sync::mpsc::Receiver;
use tracing::{info, warn};

/// Результат авторизации.
#[derive(Debug, PartialEq)]
pub enum AuthState {
    /// Готово, можно слать/принимать сообщения.
    Ready,
    /// Требуется действие пользователя (ввод кода и т.п.) — см. сообщение.
    NeedUserInput(String),
}

/// Запускает цикл авторизации. Блокирует до `ready` или ошибки.
///
/// `updates` — канал из `furunya_tg::start()` (фоновый receive-поток).
/// Возвращает `AuthState::Ready` когда сессия установлена.
pub fn authenticate(
    client: &TdClient,
    api_id: i64,
    api_hash: &str,
    database_dir: &str,
    updates: &Receiver<Value>,
) -> anyhow::Result<AuthState> {
    // 1. Параметры (плоская структура для TDLib 1.8.62).
    let params = build_td_params(api_id, api_hash, database_dir, "ru");
    client.send(&params);
    info!("[tg] setTdlibParameters отправлен, жду update-ов авторизации...");

    // Цикл обработки updateAuthorizationState через канал (не client.receive()!).
    loop {
        let v = match updates.recv_timeout(std::time::Duration::from_secs(120)) {
            Ok(v) => v,
            Err(_) => {
                anyhow::bail!("TDLib не присылает update-ы (таймаут авторизации)")
            }
        };
        if let Some(state) = handle_auth_update(&v, client)? {
            return Ok(state);
        }
    }
}

/// Обрабатывает один update. Если это updateAuthorizationState — делает действие.
/// Возвращает Some(AuthState), если авторизация завершена или требует ввод.
fn handle_auth_update(v: &Value, client: &TdClient) -> anyhow::Result<Option<AuthState>> {
    if v.get("@type").and_then(|t| t.as_str()) != Some("updateAuthorizationState") {
        return Ok(None);
    }
    let auth = &v["authorization_state"];
    let state_type = auth.get("@type").and_then(|t| t.as_str()).unwrap_or("");

    match state_type {
        "authorizationStateReady" => {
            info!("[tg] авторизация завершена (ready)");
            Ok(Some(AuthState::Ready))
        }
        "authorizationStateWaitEncryptionKey" => {
            info!("[tg] ожидание ключа шифрования БД");
            client.send(&serde_json::json!({
                "@type": "setDatabaseEncryptionKey",
                "new_encryption_key": ""
            }));
            Ok(None)
        }
        "authorizationStateWaitPhoneNumber" => {
            info!("[tg] ожидание номера телефона");
            print!("Введите номер телефона (в формате +7...): ");
            io::stdout().flush().ok();
            let mut phone = String::new();
            io::stdin().read_line(&mut phone).ok();
            client.send(&serde_json::json!({
                "@type": "setAuthenticationPhoneNumber",
                "phone_number": phone.trim()
            }));
            Ok(None)
        }
        "authorizationStateWaitCode" => {
            info!("[tg] ожидание кода подтверждения");
            print!("Введите код из Telegram: ");
            io::stdout().flush().ok();
            let mut code = String::new();
            io::stdin().read_line(&mut code).ok();
            client.send(&serde_json::json!({
                "@type": "checkAuthenticationCode",
                "code": code.trim()
            }));
            Ok(None)
        }
        "authorizationStateWaitPassword" => {
            info!("[tg] ожидание пароля 2FA");
            print!("Введите пароль 2FA: ");
            io::stdout().flush().ok();
            let mut pwd = String::new();
            io::stdin().read_line(&mut pwd).ok();
            client.send(&serde_json::json!({
                "@type": "checkAuthenticationPassword",
                "password": pwd.trim()
            }));
            Ok(None)
        }
        "authorizationStateLoggingOut" | "authorizationStateClosed" => {
            warn!("[tg] сессия закрыта/логаут");
            Ok(None)
        }
        other => {
            warn!("[tg] неизвестный auth-state: {}", other);
            Ok(None)
        }
    }
}
