//! Публичный интернет для Furinya: поиск и чтение страниц.
//!
//! Без API-ключей: DuckDuckGo (html-эндпоинт, ℎttps://html.duckduckgo.com/html/?q=)
//! + чтение произвольной страницы с грубым снятием разметки. HTTP-клиент
//! ходит через системный прокси (reqwest по умолчанию читает
//! HTTP_PROXY/HTTPS_PROXY — в LXC это Mihomo), так что блокировки не страшны.
//!
//! Не претендует на полноту Google: для задач «что сейчас происходит /
//! погода / свежая инфа по теме» этого достаточно.

use anyhow::{Context, Result};

/// Один результат поиска.
#[derive(Debug, Clone)]
pub struct WebResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

fn client() -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        // DDG душит не-браузерные UA (202-челлендж) — притворяемся Chrome-ом.
        .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
        .timeout(std::time::Duration::from_secs(18));
    // Внешний веб (Google News/DDG) часто недоступен напрямую — LXC-деплой
    // подставляет FURINYA_WEB_PROXY=socks5://127.0.0.1:7890 (Mihomo).
    if let Ok(p) = std::env::var("FURINYA_WEB_PROXY") {
        if !p.trim().is_empty() {
            builder = builder.proxy(
                reqwest::Proxy::all(p.trim()).context("невалидный FURINYA_WEB_PROXY")?,
            );
        }
    }
    Ok(builder.build().context("клиент для веба")?)
}

/// Минимальный percent-encode для query-строк (без внешних крейтов).
fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Разжимает основные HTML-сущности.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    loop {
        match rest.find('&') {
            None => {
                out.push_str(rest);
                break;
            }
            Some(i) => {
                out.push_str(&rest[..i]);
                rest = &rest[i..];
                let known = [
                    "&amp;", "&lt;", "&gt;", "&quot;", "&nbsp;", "&#39;", "&#x27;", "&#039;",
                    "&#8217;",
                ];
                let mut hit = false;
                for k in known {
                    if rest.starts_with(k) {
                        out.push_str(match k {
                            "&amp;" => "&",
                            "&lt;" => "<",
                            "&gt;" => ">",
                            "&quot;" => "\"",
                            "&nbsp;" => " ",
                            "&#8217;" => "\u{2019}",
                            // &#39;/&#x27;/&#039; — это U+0027, прямой апостроф
                            _ => "'",
                        });
                        rest = &rest[k.len()..];
                        hit = true;
                        break;
                    }
                }
                if !hit {
                    if let Some(semi) = rest[..].find(';') {
                        if semi <= 12 && rest.starts_with("&#") {
                            // числовые entity: &#1053; (десятичное) и &#x1F; (hex)
                            let code = &rest[2..semi];
                            let cp = if let Some(h) = code.strip_prefix(['x', 'X']) {
                                u32::from_str_radix(h, 16).ok()
                            } else {
                                code.parse::<u32>().ok()
                            };
                            if let Some(c) = cp.and_then(char::from_u32) {
                                out.push(c);
                                rest = &rest[semi + 1..];
                                continue;
                            }
                        }
                        // неизвестная сущность — оставляем как есть
                        out.push('&');
                        rest = &rest[1..];
                    } else {
                        out.push('&');
                        rest = &rest[1..];
                    }
                }
            }
        }
    }
    out
}

/// Вырезает HTML-теги (<b>, <span ...>) из фрагмента.
fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('<') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        match rest.find('>') {
            Some(j) => rest = &rest[j + 1..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out
}

fn clean(s: &str) -> String {
    let stripped = unescape(&strip_tags(s));
    stripped.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Выдёргивает атрибут href из фрагмента, начиная с маркера.
fn extract_href(fragment: &str) -> Option<String> {
    let i = fragment.find("href=\"")? + 6;
    let rest = &fragment[i..];
    let j = rest.find('"')?;
    Some(rest[..j].to_string())
}

/// DDG отдаёт ссылки через редирект вида
/// `//duckduckgo.com/l/?uddg=<percent-encoded>&rut=...` — разворачиваем.
fn unwrap_uddg(url: &str) -> String {
    let u = {
        let s = url.strip_prefix("//").unwrap_or(url);
        if !s.contains("duckduckgo.com/l/") {
            return url.trim().to_string();
        }
        s
    };
    let mut res = u.to_string();
    if let Some(i) = u.find("uddg=") {
        let tail = &u[i + 5..];
        let end = tail.find('&').unwrap_or(tail.len());
        if let Ok(decoded) = percent_decode(&tail[..end]) {
            res = decoded;
        }
    }
    if res.starts_with("//") {
        res = format!("https:{}", res);
    }
    res
}

fn percent_decode(s: &str) -> Result<String> {
    let bytes: Vec<u8> = {
        let mut v = Vec::with_capacity(s.len());
        let bs = s.as_bytes();
        let mut i = 0;
        while i < bs.len() {
            match bs[i] {
                b'%' if i + 2 < bs.len() => {
                    let hex = std::str::from_utf8(&bs[i + 1..i + 3])?;
                    let b = u8::from_str_radix(hex, 16)
                        .with_context(|| format!("bad hex: {hex}"))?;
                    v.push(b);
                    i += 3;
                }
                b'+' => {
                    v.push(b' ');
                    i += 1;
                }
                c => {
                    v.push(c);
                    i += 1;
                }
            }
        }
        v
    };
    String::from_utf8(bytes).context("utf8 в uddg")
}

/// Обобщённый парсерстраницы результатов DDG: ссылки по `link_marker`,
/// сниппеты по `snippet_marker` (ближайший ПОСЛЕ ссылки).
fn parse_results(html: &str, link_marker: &str, snippet_marker: &str, limit: usize) -> Vec<WebResult> {
    let mut out = Vec::new();
    let mut rest = &html[..];
    while let Some(i) = rest.find(link_marker) {
        let chunk = &rest[i..];
        let endlink = chunk.find("</a>").unwrap_or(chunk.len());
        let Some(href) = extract_href(&chunk[..endlink]) else {
            rest = &chunk[link_marker.len()..];
            continue;
        };
        // Текст ссылки: от конца открывающего <a ...> до </a>.
        let title_raw = chunk
            .find('>')
            .map(|k| &chunk[k + 1..endlink])
            .unwrap_or("");
        // Сниппет — ближайший <marker> ПОСЛЕ ссылки (до следующего результата).
        let after = &rest[i + endlink..];
        let snippet_raw = after
            .find(snippet_marker)
            .and_then(|s| {
                let seg = &after[s + snippet_marker.len()..];
                let gt = seg.find('>')?;
                let seg = &seg[gt + 1..];
                let end = ["</td>", "</span>", "</a>", "</div>"]
                    .iter()
                    .filter_map(|c| seg.find(c))
                    .min()?;
                Some(&seg[..end])
            })
            .unwrap_or("");
        let title = clean(title_raw);
        if !title.is_empty() {
            out.push(WebResult {
                url: unwrap_uddg(&href),
                title,
                snippet: clean(snippet_raw),
            });
            if out.len() >= limit.max(1) {
                break;
            }
        }
        rest = after;
    }
    out
}

/// Запрос к одному html-эндпоинту DDG с последующим парсингом.
async fn try_search(
    query_enc: &str,
    page: &str,
    link_marker: &str,
    snippet_marker: &str,
    limit: usize,
) -> Result<Vec<WebResult>> {
    let resp = client()?
        .get(page.replace("{}", query_enc))
        .send()
        .await
        .context("запрос поиска")?;
    if !resp.status().is_success() {
        // 202 = анти-бот-челлендж DDG; считаем ошибкой и падаем на фолбэк.
        anyhow::bail!("поиск {}: HTTP {}", page, resp.status());
    }
    let html = resp.text().await.context("тело поиска")?;
    Ok(parse_results(&html, link_marker, snippet_marker, limit))
}

/// Поиск через DuckDuckGo (html-эндпоинт, фолбэк — lite). Возвращает top-`limit`.
pub async fn search(query: &str, limit: usize) -> Result<Vec<WebResult>> {
    let q = url_encode(query);
    let endpoints: [(&str, &str, &str); 2] = [
        (
            "https://html.duckduckgo.com/html/?q={}&kl=ru-ru",
            "class=\"result__a\"",
            "result__snippet",
        ),
        (
            "https://lite.duckduckgo.com/lite/?q={}&kl=ru-ru",
            "class=\"result-link\"",
            "result-snippet",
        ),
    ];
    let mut last_err: anyhow::Error = anyhow::Error::msg("поиск: неизвестная ошибка");
    for (page, link_marker, snippet_marker) in endpoints {
        match try_search(&q, page, link_marker, snippet_marker, limit).await {
            Ok(list) if !list.is_empty() => return Ok(list),
            Ok(_) => {
                last_err = anyhow::Error::msg("поиск: DDG вернул пустой набор (челлендж?)")
            }
            Err(e) => last_err = e,
        }
    }
    Err(last_err)
}

/// Результат Google News RSS: заголовок, источник, дата, ссылка на редирект.
#[derive(Debug, Clone)]
pub struct NewsHeadline {
    pub title: String,
    pub source: String,
    /// «21.10 15:04» (МСК).
    pub date_str: String,
    pub url: String,
}

/// Содержимое простого XML-тега.
fn extract_tag(x: &str, tag: &str) -> Option<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let i = x.find(&open)? + open.len();
    let tail = &x[i..];
    let j = tail.find(&close)?;
    Some(tail[..j].to_string())
}

/// Разбор одного <item> Google News RSS.
fn parse_news_item(item: &str) -> Option<NewsHeadline> {
    let title_raw = extract_tag(item, "title")?;
    let title_clean = clean(&title_raw);
    if title_clean.is_empty() {
        return None;
    }
    // Титул в формате «Заголовок - Источник» — источник замыкает строку.
    let (title, source) = match title_clean.rfind(" - ") {
        Some(sep) => (
            title_clean[..sep].trim().to_string(),
            title_clean[sep + 3..].trim().to_string(),
        ),
        None => (title_clean.clone(), "Google News".to_string()),
    };
    let date_str = extract_tag(item, "pubDate")
        .and_then(|d| {
            chrono::DateTime::parse_from_rfc2822(&d)
                .ok()
                .map(|dt| {
                    dt.with_timezone(&chrono::FixedOffset::east_opt(3 * 3600).unwrap())
                        .format("%d.%m %H:%M")
                        .to_string()
                })
        })
        .unwrap_or_default();
    Some(NewsHeadline {
        title,
        source,
        date_str,
        url: extract_tag(item, "link").unwrap_or_default().trim().to_string(),
    })
}

/// Свежесть по теме через Google News RSS: без ключей, без бот-челленджей,
/// устойчивый XML. Охватывает российские СМИ (hl=ru, gl=RU).
pub async fn news_search(query: &str, limit: usize) -> Result<Vec<NewsHeadline>> {
    let url = format!(
        "https://news.google.com/rss/search?q={}&hl=ru&gl=RU&ceid=RU:ru",
        url_encode(query)
    );
    let resp = client()?.get(&url).send().await.context("Google News RSS")?;
    if !resp.status().is_success() {
        anyhow::bail!("Google News RSS: HTTP {}", resp.status());
    }
    let xml = resp.text().await.context("тело RSS")?;
    let mut out = Vec::new();
    let mut rest = &xml[..];
    while let Some(i) = rest.find("<item>") {
        let tail = &rest[i + 6..];
        let Some(end) = tail.find("</item>") else { break };
        if let Some(h) = parse_news_item(&tail[..end]) {
            out.push(h);
            if out.len() >= limit.max(1) {
                break;
            }
        }
        rest = &tail[end + 7..];
    }
    Ok(out)
}

/// Удаляет блоки <tag>...</tag> (совместно с содержимым) без разбора DOM.
fn remove_blocks(html: &str, tags: &[&str]) -> String {
    let mut out = String::with_capacity(html.len());
    let lower = html.to_ascii_lowercase();
    struct Cut<'a> {
        open_pos: usize,
        end_pos: usize,
        _t: &'a str,
    }
    // Собираем все отрезки для всех тегов, потом склеиваем внешние куски.
    let mut cuts: Vec<Cut> = Vec::new();
    for tag in tags {
        let open = format!("<{}>", tag); // script/style ездят без атрибутов
        let close = format!("</{}>", tag);
        let mut pos = 0;
        while let Some(i) = lower[pos..].find(&open) {
            let start = pos + i;
            let after_open = start + open.len();
            if let Some(j) = lower[after_open..].find(&close) {
                cuts.push(Cut {
                    open_pos: start,
                    end_pos: after_open + j + close.len(),
                    _t: tag,
                });
                pos = after_open + j + close.len();
            } else {
                break; // незакрытый блок — дальше не ищем
            }
        }
    }
    cuts.sort_by_key(|c| c.open_pos);
    let mut pos = 0;
    for c in cuts {
        if c.open_pos < pos {
            continue; // вложенный (например <style> внутри <head>? нет — head не режем)
        }
        out.push_str(&html[pos..c.open_pos]);
        pos = c.end_pos;
    }
    out.push_str(&html[pos..]);
    out
}

/// Открывает страницу и вынимает её текст (грубо, без JS) до `max_chars`.
pub async fn open_page(url: &str, max_chars: usize) -> Result<String> {
    let resp = client()?
        .get(url)
        .header("Accept", "text/html,application/xhtml+xml")
        .send()
        .await
        .with_context(|| format!("открытие {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("страница {}: HTTP {}", url, resp.status());
    }
    let html = resp.text().await.context("тело страницы")?;
    let cleaned = remove_blocks(&html, &["script", "style", "noscript"]);
    let text = clean(&cleaned);
    Ok(text.chars().take(max_chars).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_encode_basic() {
        assert_eq!(url_encode("новости nn"), "%D0%BD%D0%BE%D0%B2%D0%BE%D1%81%D1%82%D0%B8+nn");
    }

    #[test]
    fn unescape_entities() {
        assert_eq!(unescape("a&amp;b&lt;c &#1053;&#x27;d"), "a&b<c Н'd");
    }

    #[test]
    fn strip_tags_simple() {
        assert_eq!(strip_tags("<b>жирный</b> текст"), "жирный текст");
    }

    #[test]
    fn uddg_decode() {
        let url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Ft.me%2Fs%2Fnn_ru&rut=abc";
        assert_eq!(unwrap_uddg(url), "https://t.me/s/nn_ru");
    }

    #[test]
    fn news_item_parse() {
        let item = "<item><title>«Дело идёт к завершению»: встреча в Петербурге - Forbes.ru</title>\
            <link>https://news.google.com/rss/articles/CBMi?oc=5</link>\
            <pubDate>Wed, 21 Oct 2026 15:04:25 GMT</pubDate></item>";
        let h = parse_news_item(item).expect("должна разобрать");
        assert_eq!(h.source, "Forbes.ru");
        assert_eq!(h.title, "«Дело идёт к завершению»: встреча в Петербурге");
        assert_eq!(h.date_str, "21.10 18:04"); // GMT → МСК(+3)
    }
}
