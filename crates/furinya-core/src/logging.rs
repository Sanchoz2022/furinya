//! Централизованное логирование ядра Furinya.
//!
//! Инициализирует `tracing` так, чтобы всё шло одновременно:
//! - в stdout (читаемый формат для живого запуска через `cargo run`),
//! - в файл `furinya.log` (append) — чтобы при внезапном падении/краше
//!   история была доступна для разбора post-mortem.
//!
//! Thinking-chain модели пишется с тегом `thinking` на уровне debug,
//! чтобы при необходимости его можно было отфильтровать или, наоборот,
//! включить через `RUST_LOG=furinya_core=debug` / `FURINYA_LOG=debug`.

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// Инициализирует глобальный логгер.
///
/// `file_name` — куда писать файловый лог (рядом с cwd, если относительный).
/// Если файл недоступен (нет прав/диска), падает только файловый sink,
/// stdout-вывод продолжает работать.
pub fn init_logging(file_name: &str) -> anyhow::Result<()> {
    // RUST_LOG имеет приоритет; иначе дефолт — info + warn/error.
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| {
            EnvFilter::try_new("info,furinya_core=debug,furinya_llm=debug,furinya_diary=debug")
        })
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let stdout_layer = fmt::layer().with_target(false).with_writer(std::io::stdout);

    // Ротация по размеру: если лог > 5MB — переименовываем в .old и начинаем новый.
    // Это чтобы файл не рос бесконечно при частых перезапусках.
    if let Ok(meta) = std::fs::metadata(file_name)
        && meta.len() > 5 * 1024 * 1024
    {
        let _ = std::fs::rename(file_name, format!("{}.old", file_name));
    }

    // Файловый слой: тихо игнорируем ошибку открытия (оставляем только stdout).
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_name);

    match file {
        Ok(f) => {
            let file_layer = fmt::layer()
                .with_target(true)
                .with_ansi(false)
                .with_writer(f);
            tracing_subscriber::registry()
                .with(env_filter)
                .with(stdout_layer)
                .with(file_layer)
                .init();
        }
        Err(e) => {
            eprintln!(
                "[logging] не удалось открыть файл лога '{}': {e} — пишу только в stdout",
                file_name
            );
            tracing_subscriber::registry()
                .with(env_filter)
                .with(stdout_layer)
                .init();
        }
    }
    Ok(())
}

/// Грубая оценка числа токенов в строке (английский/код ~4 символа на токен,
/// для русского берём с запасом ~2.5). Возвращает примерную верхнюю оценку.
pub fn estimate_tokens(text: &str) -> usize {
    // Смесь: делим на 3 (компромисс между 4 для EN и 2.5 для RU).
    (text.chars().count() / 3).max(1)
}
