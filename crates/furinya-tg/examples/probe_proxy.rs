//! Пробник схемы прокси TDLib 1.8.62.
//!
//! В этой версии класса "proxySettings" нет (ответ: Unknown class), а addProxy
//! ждёт параметр "proxy" — вложенный объект. Пробуем несколько форм и печатаем
//! ответы TDLib, чтобы понять точную схему (и тип опции active_proxy_id).
//!
//! Запуск: cargo run --example probe_proxy

use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = furinya_core::config::Config::load(std::path::Path::new("config.toml"))?;
    let (client, update_loop) = furinya_tg::TdClient::start(
        furinya_tg::default_lib_path().to_str().unwrap_or("libs/libtdjson.so"),
        cfg.general.telegram_api_id.into(),
        &cfg.general.telegram_api_hash,
        "/tmp/probe_proxy_db",
    )?;
    let params = furinya_tg::build_td_params(
        cfg.general.telegram_api_id.into(),
        &cfg.general.telegram_api_hash,
        "/tmp/probe_proxy_db",
        "ru",
    );
    client.send(&params);
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Список прокси (формат объекта) и тип опции.
    for req in [
        serde_json::json!({"@type": "getProxies"}),
        serde_json::json!({"@type": "getOption", "name": "active_proxy_id"}),
        serde_json::json!({"@type": "getOption", "name": "use_proxy"}),
        serde_json::json!({"@type": "getOption", "name": "use_proxy_for_calls"}),
    ] {
        let extra = format!("probe_{}", req.to_string());
        match client.query_td(req.clone(), &extra, Duration::from_secs(15)).await {
            Ok(v) => println!("== {}\n   → {}\n", req, v),
            Err(e) => println!("== {}\n   → ОШИБКА {}\n", req, e),
        }
    }

    // Кандидаты addProxy.
    let candidates = vec![
        serde_json::json!({
            "@type": "addProxy",
            "proxy": {
                "server": "127.0.0.1", "port": 7890, "is_enabled": true,
                "type": {"@type": "proxyTypeSocks5"}
            }
        }),
        serde_json::json!({
            "@type": "addProxy",
            "proxy": {
                "server": "127.0.0.2", "port": 7890, "enable": true,
                "type": {"@type": "proxyTypeSocks5"}
            }
        }),
    ];
    let mut proxy_id: Option<i64> = None;
    for (i, c) in candidates.iter().enumerate() {
        let extra = format!("probe_add_{i}");
        match client.query_td(c.clone(), &extra, Duration::from_secs(15)).await {
            Ok(v) => {
                println!("== кандидат {i}\n   → {}\n", v);
                if proxy_id.is_none() {
                    proxy_id = v.get("id").and_then(|x| x.as_i64());
                }
            }
            Err(e) => println!("== кандидат {i}\n   → ОШИБКА {}\n", e),
        }
    }

    // Включение прокси: пробуем разные опции/методы.
    if let Some(id) = proxy_id {
        println!("\n=== включаем прокси id={id} ===");
        for (name, val) in [
            ("use_proxy", serde_json::json!({"@type": "optionValueInteger", "value": id})),
            (
                "use_proxy_for_calls",
                serde_json::json!({"@type": "optionValueBoolean", "value": true}),
            ),
            (
                "active_proxy_id",
                serde_json::json!({"@type": "optionValueInteger", "value": id}),
            ),
        ] {
            let extra = format!("probe_set_{name}");
            let req = serde_json::json!({"@type": "setOption", "name": name, "value": val});
            match client.query_td(req.clone(), &extra, Duration::from_secs(15)).await {
                Ok(v) => println!("== setOption {name} → {}\n", v),
                Err(e) => println!("== setOption {name} → ОШИБКА {}\n", e),
            }
        }
        // editProxy с is_enabled
        let extra = format!("probe_edit");
        let req = serde_json::json!({
            "@type": "editProxy", "id": id, "is_enabled": true
        });
        match client.query_td(req.clone(), &extra, Duration::from_secs(15)).await {
            Ok(v) => println!("== editProxy → {}\n", v),
            Err(e) => println!("== editProxy → ОШИБКА {}\n", e),
        }
        // enableProxy — включаем прокси
        for (label, req) in [
            (
                "enableProxy id",
                serde_json::json!({"@type": "enableProxy", "id": id}),
            ),
            (
                "enableProxy proxy_id",
                serde_json::json!({"@type": "enableProxy", "proxy_id": id}),
            ),
        ] {
            let extra = format!("probe_enable_{label}");
            match client.query_td(req.clone(), &extra, Duration::from_secs(15)).await {
                Ok(v) => println!("== {label} → {}\n", v),
                Err(e) => println!("== {label} → ОШИБКА {}\n", e),
            }
        }
        // итоговое состояние
        let extra = format!("probe_getproxies2");
        match client
            .query_td(serde_json::json!({"@type": "getProxies"}), &extra, Duration::from_secs(15))
            .await
        {
            Ok(v) => println!("== getProxies (итог) → {}\n", v),
            Err(e) => println!("== getProxies → ОШИБКА {}\n", e),
        }
        for name in ["use_proxy", "use_proxy_for_calls"] {
            let extra = format!("probe_final_{name}");
            match client
                .query_td(
                    serde_json::json!({"@type": "getOption", "name": name}),
                    &extra,
                    Duration::from_secs(15),
                )
                .await
            {
                Ok(v) => println!("== getOption {name} (итог) → {}\n", v),
                Err(e) => println!("== getOption {name} → ОШИБКА {}\n", e),
            }
        }
    }
    let _ = update_loop;
    Ok(())
}
