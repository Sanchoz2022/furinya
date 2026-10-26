//! Идентификация собеседника по аккаунту Telegram, а не по имени в коде.
//!
//! Раньше все не-свои сообщения подписывались `papik_name` из конфига (или
//! `user<id>`), а проверка «это владелец?» была сравнением с `papik_chat_id`.
//! Имя в коде («Саша»/«Александр») при этом ничего не значит: аккаунт может
//! сменить first_name, писать с другого username или вообще быть не тем.
//!
//! Здесь единая точка правды:
//! - **кто владелец** — по `papik_chat_id` (основной признак) ИЛИ по
//!   `papik_username` (регистронезависимо, без `@`);
//! - **как подписать говорящего** — по live-профилю TDLib (first_name,
//!   last_name, @username), который кэшируется из `updateUser`/`getUser`.
//!
//! Кэш process-wide: сообщения обрабатываются одним циклом, поэтому гонок
//! нет; `Mutex` нужен только потому, что `resolve` может зваться из разных
//! мест (handle_new_message, bootstrap, history load).

use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tracing::debug;

use furinya_core::config::Config;

use crate::TdClient;

/// Профиль Telegram-пользователя, минимально нужный для подписи реплик.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct UserProfile {
    pub id: i64,
    pub first_name: String,
    pub last_name: String,
    /// Основной @username (без `@`), если у аккаунта он есть.
    pub username: Option<String>,
}

impl UserProfile {
    /// Разбирает объект TDLib `user` (ответ `getUser` / вложенный `updateUser.user`).
    ///
    /// Поддерживаем оба формата username: новый `usernames.active_usernames`
    /// (TDLib >= 1.8.30) и старый плоский `username`.
    pub fn from_td(v: &Value) -> Option<Self> {
        let id = v.get("id").and_then(Value::as_i64)?;
        let first_name = v
            .get("first_name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let last_name = v
            .get("last_name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string();
        let username = v
            .get("usernames")
            .and_then(|u| u.get("active_usernames"))
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.trim_start_matches('@').to_string())
            .or_else(|| {
                v.get("username")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| s.trim_start_matches('@').to_string())
            });
        Some(Self {
            id,
            first_name,
            last_name,
            username,
        })
    }

    /// Человекочитаемая подпись: «Имя Фамилия (@username)», «@username» или
    /// `user<id>` в крайнем случае. Имя берётся из аккаунта, не из конфига.
    pub fn display(&self) -> String {
        let full = [self.first_name.as_str(), self.last_name.as_str()]
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        match (&full[..], self.username.as_deref()) {
            ("", Some(u)) => format!("@{u}"),
            ("", None) => format!("user{}", self.id),
            (name, Some(u)) => format!("{name} (@{u})"),
            (name, None) => name.to_string(),
        }
    }
}

fn cache() -> &'static Mutex<HashMap<i64, UserProfile>> {
    static USERS: OnceLock<Mutex<HashMap<i64, UserProfile>>> = OnceLock::new();
    USERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Кладёт профиль в кэш (идемпотентно).
pub fn remember(profile: UserProfile) {
    cache().lock().unwrap().insert(profile.id, profile);
}

/// Обновляет кэш из TDLib update-а (`updateUser`) или ответа (`user`).
/// Возвращает true, если профиль был распознан.
pub fn remember_from_update(v: &Value) -> bool {
    let profile = match v.get("@type").and_then(Value::as_str) {
        Some("updateUser") => v.get("user").and_then(UserProfile::from_td),
        Some("user") => UserProfile::from_td(v),
        _ => None,
    };
    match profile {
        Some(p) => {
            remember(p);
            true
        }
        None => false,
    }
}

/// Профиль из кэша, если он уже пришёл.
pub fn cached(user_id: i64) -> Option<UserProfile> {
    cache().lock().unwrap().get(&user_id).cloned()
}

/// Владелец ли это: по user_id ИЛИ по username аккаунта.
///
/// `user_id` — основной признак (в приватном чате chat_id == user_id).
/// `username` — запасной: нужен, если владелец пишет с другого аккаунта
/// или его id в конфиге устарел. `papik_username` сравнивается без `@`
/// и без учёта регистра.
pub fn is_owner(cfg: &Config, user_id: i64, username: Option<&str>) -> bool {
    let owner_id = cfg.general.papik_chat_id;
    if user_id != 0 && owner_id != 0 && user_id == owner_id {
        return true;
    }
    match (cfg.general.papik_username_clean(), username) {
        (Some(want), Some(got)) => got
            .trim()
            .trim_start_matches('@')
            .eq_ignore_ascii_case(want),
        _ => false,
    }
}

/// Имя владельца для логов/промптов. Если `papik_name` не задан — берём
/// live-профиль из кэша, иначе username из конфига, иначе «собеседник».
pub fn owner_display(cfg: &Config) -> String {
    if !cfg.general.papik_name.trim().is_empty() {
        return cfg.general.papik_name.trim().to_string();
    }
    if let Some(profile) = cached(cfg.general.papik_chat_id) {
        let display = profile.display();
        if !display.starts_with("user") {
            return display;
        }
    }
    cfg.general.papik_display_name()
}

/// Подпись автора сообщения: персонаж, владелец или аккаунт из кэша.
///
/// `0` — служебный sender (idle-стимул) — возвращает пустую строку.
pub fn display_name(cfg: &Config, my_id: i64, sender: i64) -> String {
    if sender == 0 {
        return String::new();
    }
    if my_id != 0 && sender == my_id {
        return cfg.general.character_name.clone();
    }
    let profile = cached(sender);
    let username = profile.as_ref().and_then(|p| p.username.as_deref());
    if is_owner(cfg, sender, username) {
        return owner_display(cfg);
    }
    match profile {
        Some(p) => p.display(),
        None => format!("user{sender}"),
    }
}

/// Дозапрашивает профиль через `getUser` и кэширует его.
///
/// Вызывать перед подписью сообщения: TDLib не гарантирует, что `updateUser`
/// для отправителя пришёл раньше `updateNewMessage` (особенно на истории).
pub async fn resolve(client: &TdClient, user_id: i64) -> Option<UserProfile> {
    if user_id == 0 {
        return None;
    }
    if let Some(profile) = cached(user_id) {
        return Some(profile);
    }
    let extra = format!("getuser_{user_id}");
    match client
        .query_td(
            serde_json::json!({ "@type": "getUser", "user_id": user_id }),
            &extra,
            Duration::from_secs(8),
        )
        .await
    {
        Ok(v) => match UserProfile::from_td(&v) {
            Some(profile) => {
                debug!("[tg] identity: user {} -> {}", user_id, profile.display());
                remember(profile.clone());
                Some(profile)
            }
            None => {
                debug!("[tg] identity: getUser({user_id}) вернул не user: {v}");
                None
            }
        },
        Err(e) => {
            debug!("[tg] identity: getUser({user_id}) не удался: {e}");
            None
        }
    }
}

/// Резолвит пачку id (дубли и `0` отбрасываются). Последовательно, потому что
/// сообщений за раз немного, а параллельные `getUser` с одинаковым `@extra`
/// могут перетереть друг друга.
pub async fn resolve_many(client: &TdClient, ids: impl IntoIterator<Item = i64>) {
    let mut seen = std::collections::HashSet::new();
    for id in ids {
        if id == 0 || !seen.insert(id) {
            continue;
        }
        let _ = resolve(client, id).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with(name: &str, username: &str, owner_id: i64) -> Config {
        let mut cfg = Config {
            general: Default::default(),
            misc: Default::default(),
            capabilities: Default::default(),
        };
        cfg.general.papik_name = name.to_string();
        cfg.general.papik_username = username.to_string();
        cfg.general.papik_chat_id = owner_id;
        cfg.general.character_name = "Furinya".to_string();
        cfg
    }

    #[test]
    fn parses_new_tdlib_usernames_object() {
        let v = serde_json::json!({
            "@type": "user",
            "id": 42,
            "first_name": "Иван",
            "last_name": "Петров",
            "usernames": { "active_usernames": ["ivan_p"], "editable_username": "ivan_p" }
        });
        let p = UserProfile::from_td(&v).expect("profile");
        assert_eq!(p.id, 42);
        assert_eq!(p.username.as_deref(), Some("ivan_p"));
        assert_eq!(p.display(), "Иван Петров (@ivan_p)");
    }

    #[test]
    fn parses_legacy_flat_username() {
        let v = serde_json::json!({
            "@type": "user",
            "id": 7,
            "first_name": "",
            "last_name": "",
            "username": "@old_style"
        });
        let p = UserProfile::from_td(&v).expect("profile");
        assert_eq!(p.username.as_deref(), Some("old_style"));
        assert_eq!(p.display(), "@old_style");
    }

    #[test]
    fn owner_is_matched_by_id() {
        let cfg = cfg_with("", "", 5005580127);
        assert!(is_owner(&cfg, 5005580127, None));
        assert!(!is_owner(&cfg, 111, None));
    }

    #[test]
    fn owner_is_matched_by_username_case_insensitively() {
        let cfg = cfg_with("", "@Some_Owner", 0);
        assert!(is_owner(&cfg, 111, Some("@some_owner")));
        assert!(is_owner(&cfg, 111, Some("SOME_OWNER")));
        assert!(!is_owner(&cfg, 111, Some("other")));
        assert!(!is_owner(&cfg, 111, None));
    }

    #[test]
    fn display_name_uses_account_not_hardcoded_person() {
        let cfg = cfg_with("", "", 5005580127);
        remember(UserProfile {
            id: 5005580127,
            first_name: "Реальное".into(),
            last_name: "Имя".into(),
            username: Some("real_owner".into()),
        });
        remember(UserProfile {
            id: 999,
            first_name: "Гость".into(),
            last_name: String::new(),
            username: Some("guest".into()),
        });

        // Свои сообщения — имя персонажа.
        assert_eq!(display_name(&cfg, 8337379413, 8337379413), "Furinya");
        // Владелец подписан live-именем аккаунта, а не «Александр».
        assert_eq!(
            display_name(&cfg, 8337379413, 5005580127),
            "Реальное Имя (@real_owner)"
        );
        // Посторонний — своим аккаунтом.
        assert_eq!(display_name(&cfg, 8337379413, 999), "Гость (@guest)");
        // Неизвестный id — стабильный fallback.
        assert_eq!(display_name(&cfg, 8337379413, 1234), "user1234");
        // Служебный idle-sender не даёт «user0».
        assert_eq!(display_name(&cfg, 8337379413, 0), "");
    }

    #[test]
    fn explicit_papik_name_wins_for_owner_label() {
        // Отдельный id, чтобы не делить кэш с другим тестом (тесты идут параллельно).
        let cfg = cfg_with("Алекс", "", 777001);
        remember(UserProfile {
            id: 777001,
            first_name: "Реальное".into(),
            last_name: String::new(),
            username: Some("real_owner".into()),
        });
        assert_eq!(display_name(&cfg, 8337379413, 777001), "Алекс");
    }

    #[test]
    fn remember_from_update_accepts_both_shapes() {
        let cfg = cfg_with("", "", 5);
        let upd = serde_json::json!({
            "@type": "updateUser",
            "user": { "id": 5, "first_name": "Пять", "last_name": "", "username": "five" }
        });
        assert!(remember_from_update(&upd));
        assert_eq!(display_name(&cfg, 0, 5), "Пять (@five)");

        let reply = serde_json::json!({
            "@type": "user", "id": 6, "first_name": "", "last_name": "",
            "usernames": { "active_usernames": ["six"] }
        });
        assert!(remember_from_update(&reply));
        assert_eq!(cached(6).and_then(|p| p.username), Some("six".to_string()));
        assert!(!remember_from_update(
            &serde_json::json!({"@type": "updateNewMessage"})
        ));
    }
}
