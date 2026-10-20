//! Деплой Furinya в LXC на Proxmox — целиком на Rust, без bash-скриптов.
//!
//! Что делает:
//!   1. собирает релизный бинарь (`cargo build --release -p furinya`);
//!   2. собирает payload: бинарь, libs/libtdjson.so (со стрипом), prompts/,
//!      config.toml, data/tdlib_db (сессия Telegram), data/diary (память),
//!      .env с ключами из env, systemd-юниты и клиент Mihomo;
//!   3. по SSH (ssh2) создаёт/поднимает LXC на Proxmox, заливает payload
//!      через SFTP → `pct push` → `pct exec`, ставит systemd-юниты;
//!   4. проверяет, что сервис живой, и печатает логи.
//!
//! Запуск:
//!   cargo run --release -p furinya-deploy -- \
//!       --host 192.168.1.241 --key ~/.ssh/id_deploy_pve --lxc-id 111

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use ssh2::{Channel, Session};
use std::fs;
use std::io::Write as _;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const APP_DIR: &str = "/opt/furinya";

// ─────────────────────────── аргументы ───────────────────────────

#[derive(Debug, Clone)]
struct Args {
    host: String,
    port: u16,
    user: String,
    key: PathBuf,
    lxc_id: u32,
    lxc_name: String,
    lxc_ip: String,
    bridge: String,
    gateway: String,
    cores: u32,
    mem_mb: u32,
    disk_gb: u32,
    template: Option<String>,
    tz: String,
    build: bool,
    with_mihomo: bool,
    project_root: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            host: "192.168.1.241".into(),
            port: 22,
            user: "root".into(),
            key: default_key(),
            lxc_id: 111,
            lxc_name: "furinya".into(),
            lxc_ip: "192.168.1.75".into(),
            bridge: "vmbr0".into(),
            gateway: "192.168.1.1".into(),
            cores: 2,
            mem_mb: 2048,
            disk_gb: 12,
            template: None,
            tz: "Europe/Moscow".into(),
            build: true,
            with_mihomo: true,
            project_root: project_root(),
        }
    }
}

fn project_root() -> PathBuf {
    // crates лежат в <root>/crates/..., бинари — в <root>/bins/...
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or(manifest)
}

fn default_key() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
    PathBuf::from(home).join(".ssh/id_deploy_pve")
}

fn parse_args() -> Result<Args> {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].clone();
        let take = |i: &mut usize| -> Result<String> {
            *i += 1;
            argv.get(*i)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("после {key} нужен значение"))
        };
        match argv[i].as_str() {
            "--host" => a.host = take(&mut i)?,
            "--port" => a.port = take(&mut i)?.parse()?,
            "--user" => a.user = take(&mut i)?,
            "--key" => a.key = PathBuf::from(take(&mut i)?),
            "--lxc-id" => a.lxc_id = take(&mut i)?.parse()?,
            "--lxc-name" => a.lxc_name = take(&mut i)?,
            "--lxc-ip" => a.lxc_ip = take(&mut i)?,
            "--bridge" => a.bridge = take(&mut i)?,
            "--gateway" => a.gateway = take(&mut i)?,
            "--cores" => a.cores = take(&mut i)?.parse()?,
            "--mem" => a.mem_mb = take(&mut i)?.parse()?,
            "--disk" => a.disk_gb = take(&mut i)?.parse()?,
            "--template" => a.template = Some(take(&mut i)?),
            "--tz" => a.tz = take(&mut i)?,
            "--no-build" => a.build = false,
            "--no-mihomo" => a.with_mihomo = false,
            "--root" => a.project_root = PathBuf::from(take(&mut i)?),
            other => bail!("неизвестный аргумент {other}"),
        }
        i += 1;
    }
    Ok(a)
}

// ─────────────────────────── SSH-обвязка ───────────────────────────

struct Remote {
    session: Session,
}

impl Remote {
    fn connect(a: &Args) -> Result<Remote> {
        let tcp = TcpStream::connect((a.host.as_str(), a.port))
            .with_context(|| format!("подключение к {}:{}", a.host, a.port))?;
        let mut session = Session::new().context("SSH-сессия")?;
        session.set_tcp_stream(tcp);
        session.set_blocking(true);
        session.handshake().context("SSH handshake")?;
        session
            .userauth_pubkey_file(&a.user, None, &a.key, None)
            .with_context(|| format!("авторизация ключом {}", a.key.display()))?;
        Ok(Remote { session })
    }

    /// Выполняет команду, возвращает (код возврата, stdout+stderr).
    fn run(&self, cmd: &str) -> Result<(i32, String)> {
        let mut channel: Channel = self.session.channel_session().context("ssh channel")?;
        channel.exec(cmd).context("exec")?;
        let mut out = String::new();
        let mut buf = [0u8; 8192];
        {
            let mut stdout = channel.stream(0);
            loop {
                match std::io::Read::read(&mut stdout, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => out.push_str(&String::from_utf8_lossy(&buf[..n])),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                    Err(e) => bail!("чтение stdout: {e}"),
                }
            }
        }
        {
            let mut stderr = channel.stderr();
            loop {
                match std::io::Read::read(&mut stderr, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => out.push_str(&String::from_utf8_lossy(&buf[..n])),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
                    Err(e) => bail!("чтение stderr: {e}"),
                }
            }
        }
        channel.wait_close().context("wait_close")?;
        let code = channel.exit_status().unwrap_or(-1);
        Ok((code, out))
    }

    /// Выполняет команду; при ненулевом коде — ошибка с выводом.
    fn must(&self, cmd: &str) -> Result<String> {
        let (code, out) = self.run(cmd)?;
        if code != 0 {
            bail!("команда упала (код {code}): {cmd}\n{out}");
        }
        Ok(out)
    }

    /// Заливает файл на хост через SFTP.
    fn upload(&self, local: &Path, remote: &str) -> Result<()> {
        let sftp = self.session.sftp().context("sftp")?;
        let data = fs::read(local)
            .with_context(|| format!("чтение {}", local.display()))?;
        let mut f: ssh2::File = sftp
            .open_mode(
                Path::new(remote),
                ssh2::OpenFlags::WRITE | ssh2::OpenFlags::TRUNCATE | ssh2::OpenFlags::CREATE,
                0o644,
                ssh2::OpenType::File,
            )
            .with_context(|| format!("sftp open {remote}"))?;
        f.write_all(&data).context("sftp write")?;
        Ok(())
    }
}

// ─────────────────────────── payload ───────────────────────────

/// Параметры VLESS-узла для Mihomo (deploy/secrets/node.json).
#[derive(Debug, Deserialize)]
struct VlessNode {
    server: String,
    server_port: u16,
    uuid: String,
    #[serde(default)]
    tls: TlsCfg,
    #[serde(default)]
    transport: TransportCfg,
}

#[derive(Debug, Default, Deserialize)]
struct TlsCfg {
    #[serde(default)]
    server_name: String,
    #[serde(default)]
    utls: UtlsCfg,
}

#[derive(Debug, Default, Deserialize)]
struct UtlsCfg {
    #[serde(default)]
    fingerprint: String,
}

#[derive(Debug, Default, Deserialize)]
struct TransportCfg {
    #[serde(default)]
    path: String,
}

fn render_mihomo_config(node: &VlessNode, mix_port: u16) -> String {
    let sni = if node.tls.server_name.is_empty() {
        node.server.clone()
    } else {
        node.tls.server_name.clone()
    };
    let fp = if node.tls.utls.fingerprint.is_empty() {
        "chrome".to_string()
    } else {
        node.tls.utls.fingerprint.clone()
    };
    let path = if node.transport.path.is_empty() {
        "/".to_string()
    } else {
        node.transport.path.clone()
    };
    format!(
        "# Сгенерировано furinya-deploy. Прокси для внешних API и Telegram.
mixed-port: {mix_port}
allow-lan: false
bind-address: 127.0.0.1
mode: rule
log-level: info
ipv6: false
unified-delay: true
tcp-concurrent: true

proxies:
  - name: vless-main
    type: vless
    server: {server}
    port: {port}
    uuid: {uuid}
    network: ws
    tls: true
    servername: {sni}
    client-fingerprint: {fp}
    skip-cert-verify: false
    ws-opts:
      path: {path}
      headers:
        Host: {sni}

proxy-groups:
  - name: PROXY
    type: select
    proxies:
      - vless-main

rules:
  - GEOIP,private,DIRECT
  - MATCH,PROXY
",
        server = node.server,
        port = node.server_port,
        uuid = node.uuid,
        sni = sni,
        fp = fp,
        path = path,
        mix_port = mix_port,
    )
}

const MIHOMO_SERVICE: &str = r#"[Unit]
Description=Mihomo proxy client (внешние API + Telegram)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/mihomo -d /etc/mihomo
Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
"#;

/// Собирает tar-архив payload в `stage`.
fn build_payload(a: &Args, stage: &Path) -> Result<()> {
    fs::create_dir_all(stage.join("payload/data"))?;
    let root = &a.project_root;

    // 1) бинарь
    let bin = a.project_root.join("target-local/release/furinya");
    if !bin.exists() {
        bail!(
            "нет {} — собери релиз (cargo build --release -p furinya)",
            bin.display()
        );
    }
    fs::copy(&bin, stage.join("payload/furinya"))?;
    set_exec(&stage.join("payload/furinya"))?;

    // 2) libtdjson.so (830 МБ → стрип debug-символов)
    let so = root.join("libs/libtdjson.so");
    if so.exists() {
        let dst = stage.join("payload/libs/libtdjson.so");
        fs::create_dir_all(dst.parent().unwrap())?;
        fs::copy(&so, &dst)?;
        strip_binary(&dst);
        println!("  libtdjson.so: {}", human_size(fs::metadata(&dst)?.len()));
    } else {
        println!("  ⚠ нет libs/libtdjson.so — генератор фото работать не будет");
    }

    // 3) промпты
    let prompts_dir = stage.join("payload/prompts");
    fs::create_dir_all(&prompts_dir)?;
    for entry in fs::read_dir(root.join("prompts"))? {
        let p = entry?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("md") {
            fs::copy(&p, prompts_dir.join(p.file_name().unwrap()))?;
        }
    }

    // 4) конфиг (реальный, с ключами из env)
    let cfg = root.join("config.toml");
    if cfg.exists() {
        fs::copy(&cfg, stage.join("payload/config.toml"))?;
    }

    // 5) сессия Telegram + память
    for (src, dst) in [
        ("data/tdlib_db", "payload/data/tdlib_db"),
        ("data/diary", "payload/data/diary"),
    ] {
        let s = root.join(src);
        if s.exists() {
            copy_dir_all(&s, &stage.join(dst))?;
        }
    }

    // 6) .env с ключами (значения не печатаются)
    let env = format!(
        "MISTRAL_API_KEY={}\nGROQ_API_KEY={}\nFISH_API_KEY={}\nFURINYA_HORDE_KEY={}\nSD_API_URL={}\nFURINYA_TG_PROXY=socks5://127.0.0.1:7890\nFURINYA_TD_QUIET=1\n",
        std::env::var("MISTRAL_API_KEY").unwrap_or_default(),
        std::env::var("GROQ_API_KEY").unwrap_or_default(),
        std::env::var("FISH_API_KEY").unwrap_or_default(),
        std::env::var("FURINYA_HORDE_KEY").unwrap_or_default(),
        std::env::var("SD_API_URL").unwrap_or_else(|_| "http://192.168.1.172:8082/".into()),
    );
    fs::write(stage.join("payload/.env"), env)?;

    // 7) systemd-юнит бота
    let unit = root.join("deploy/furinya.service");
    if unit.exists() {
        fs::copy(&unit, stage.join("payload/furinya.service"))?;
    }

    // 8) Mihomo: конфиг из node.json + юнит + сам бинарь
    if a.with_mihomo {
        let node_path = root.join("deploy/secrets/node.json");
        if node_path.exists() {
            let node: VlessNode =
                serde_json::from_str(&fs::read_to_string(&node_path)?).context("node.json")?;
            fs::write(
                stage.join("payload/mihomo.yaml"),
                render_mihomo_config(&node, 7890),
            )?;
            fs::write(stage.join("payload/mihomo.service"), MIHOMO_SERVICE)?;
            if let Some(bin) = download_mihomo()? {
                fs::copy(&bin, stage.join("payload/mihomo"))?;
                set_exec(&stage.join("payload/mihomo"))?;
                println!("  mihomo: скачан ({})", human_size(fs::metadata(&bin)?.len()));
            }
        } else {
            println!("  ⚠ нет deploy/secrets/node.json — Mihomo пропущен");
        }
    }
    Ok(())
}

fn set_exec(p: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(p)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(p, perms)?;
    }
    Ok(())
}

fn strip_binary(p: &Path) {
    let out = std::process::Command::new("strip")
        .arg("--strip-debug")
        .arg(p)
        .output();
    if let Ok(o) = out {
        if !o.status.success() {
            println!("  ⚠ strip не сработал: {}", String::from_utf8_lossy(&o.stderr));
        }
    }
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let e = entry?;
        let from = e.path();
        let to = dst.join(e.file_name());
        if from.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            fs::copy(&from, &to).ok();
        }
    }
    Ok(())
}

fn human_size(n: u64) -> String {
    const MB: f64 = 1_048_576.0;
    if n as f64 > MB {
        format!("{:.1} МБ", n as f64 / MB)
    } else {
        format!("{:.0} КБ", n as f64 / 1024.0)
    }
}

/// Качает последний релиз Mihomo (linux amd64) — возвращает путь к бинарю.
/// Если не вышло — Ok(None): на сервере уже установленный mihomo не трогаем.
fn download_mihomo() -> Result<Option<PathBuf>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(120))
        .user_agent("furinya-deploy")
        .build()?;
    let resp = client
        .get("https://api.github.com/repos/MetaCubeX/mihomo/releases/latest")
        .send()?;
    if !resp.status().is_success() {
        println!("  ⚠ GitHub API: {} — mihomo пропущен", resp.status());
        return Ok(None);
    }
    let body = resp.text()?;
    let rel: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            println!("  ⚠ не разобрал ответ GitHub ({e}) — mihomo пропущен");
            return Ok(None);
        }
    };
    let tag = match rel["tag_name"].as_str() {
        Some(t) => t.to_string(),
        None => {
            println!("  ⚠ нет tag_name в ответе GitHub — mihomo пропущен");
            return Ok(None);
        }
    };
    let assets = match rel["assets"].as_array() {
        Some(a) => a,
        None => return Ok(None),
    };
    let name = assets
        .iter()
        .filter_map(|a| a["name"].as_str())
        .find(|n| n.starts_with("mihomo-linux-amd64-v1-") && n.ends_with(".gz"))
        .or_else(|| {
            assets
                .iter()
                .filter_map(|a| a["name"].as_str())
                .find(|n| n.starts_with("mihomo-linux-amd64-compatible-") && n.ends_with(".gz"))
        });
    let name = match name {
        Some(n) => n.to_string(),
        None => {
            println!("  ⚠ нет linux-amd64 ассета в {tag} — mihomo пропущен");
            return Ok(None);
        }
    };
    let url = format!("https://github.com/MetaCubeX/mihomo/releases/download/{tag}/{name}");
    println!("  mihomo: качаю {name}");
    let bytes = client.get(&url).send()?.bytes()?;
    let mut dec = flate2::read::GzDecoder::new(&bytes[..]);
    let mut out = Vec::new();
    std::io::Read::read_to_end(&mut dec, &mut out)?;
    let tmp = std::env::temp_dir().join("mihomo-bin");
    fs::write(&tmp, out)?;
    Ok(Some(tmp))
}

fn make_tar(stage: &Path, out: &Path) -> Result<()> {
    let f = fs::File::create(out)?;
    let mut ar = tar::Builder::new(f);
    // Записи без верхнего каталога — распаковка идёт прямо в /opt/furinya.
    ar.append_dir_all(".", stage.join("payload"))?;
    ar.finish()?;
    Ok(())
}

// ─────────────────────────── основной сценарий ───────────────────────────

fn main() -> Result<()> {
    let a = parse_args()?;
    println!("═══ Деплой Furinya → {} (LXC {}) ═══", a.host, a.lxc_id);

    // 1) сборка бинаря
    // Отдельный target-каталог: в `target/` могут лежать файлы, собранные
    // в docker-контейнере от root — cargo туда не может писать.
    let rel_target = a.project_root.join("target-local");
    if a.build {        println!("→ cargo build --release -p furinya (target: target-local)");
        let out = std::process::Command::new("cargo")
            .args(["build", "--release", "-p", "furinya"])
            .current_dir(&a.project_root)
            .env("CARGO_TARGET_DIR", &rel_target)
            .status()
            .context("cargo build")?;
        if !out.success() {
            bail!("cargo build упал");
        }
    }

    // 2) payload
    println!("→ собираю payload");
    let stage = std::env::temp_dir().join("furinya-deploy-stage");
    let _ = fs::remove_dir_all(&stage);
    fs::create_dir_all(&stage)?;
    build_payload(&a, &stage)?;
    let tar_path = std::env::temp_dir().join("furinya-payload.tar");
    make_tar(&stage, &tar_path)?;
    let tar_size = fs::metadata(&tar_path)?.len();
    println!("  payload: {}", human_size(tar_size));

    // 3) SSH
    println!("→ SSH {}", a.host);
    let remote = Remote::connect(&a)?;

    // 4) LXC
    ensure_lxc(&remote, &a)?;

    // 5) заливка
    println!("→ заливаю payload ({} МБ)", tar_size as f64 / 1_048_576.0);
    remote.upload(&tar_path, "/tmp/furinya-payload.tar")?;
    remote.must(&format!(
        "pct push {} /tmp/furinya-payload.tar /tmp/payload.tar",
        a.lxc_id
    ))?;
    remote.must(&format!(
        "pct exec {} -- bash -c 'rm -rf {APP_DIR} && mkdir -p {APP_DIR} && tar xf /tmp/payload.tar -C {APP_DIR} && rm -f /tmp/payload.tar'",
        a.lxc_id
    ))?;

    // 6) настройка и запуск
    println!("→ ставлю сервисы");
    let setup = format!(
        r#"
set -e
id furinya >/dev/null 2>&1 || useradd -m -s /bin/bash furinya
chown -R furinya:furinya {APP_DIR}
chmod 600 {APP_DIR}/.env
chmod 700 {APP_DIR}/data/tdlib_db 2>/dev/null || true
install -m 644 {APP_DIR}/furinya.service /etc/systemd/system/furinya.service
if [ -f {APP_DIR}/mihomo.service ]; then
  mkdir -p /etc/mihomo
  install -m 600 {APP_DIR}/mihomo.yaml /etc/mihomo/config.yaml
  install -m 644 {APP_DIR}/mihomo.service /etc/systemd/system/mihomo.service
  install -m 755 {APP_DIR}/mihomo /usr/local/bin/mihomo
  systemctl daemon-reload
  systemctl enable --now mihomo
  sleep 3
fi
systemctl daemon-reload
systemctl enable --now furinya
"#,
        APP_DIR = APP_DIR
    );
    remote.must(&format!(
        "pct exec {} -- bash -c '{}'",
        a.lxc_id,
        setup.replace('\'', "'\\''")
    ))?;

    // 7) проверка
    println!("→ проверяю");
    std::thread::sleep(Duration::from_secs(8));
    let (code, out) = remote.run(&format!(
        "pct exec {} -- bash -c 'systemctl is-active furinya; systemctl is-active mihomo 2>/dev/null || true; journalctl -u furinya -n 12 --no-pager'",
        a.lxc_id
    ))?;
    println!("{out}");
    if code != 0 {
        bail!("проверка не прошла");
    }
    let _ = fs::remove_dir_all(&stage);
    println!("\n✔ Готово. Логи: ssh {user}@{host} 'pct exec {id} -- journalctl -u furinya -f'",
        user = a.user, host = a.host, id = a.lxc_id);
    println!("  Локальный бот держи выключенным — Telegram-сессия одна.");
    Ok(())
}

/// Создаёт LXC (если нет), скачивает шаблон, поднимает и ждёт сеть.
fn ensure_lxc(remote: &Remote, a: &Args) -> Result<()> {
    let exists = remote
        .run(&format!("pct status {} >/dev/null 2>&1 && echo yes || echo no", a.lxc_id))?
        .1;
    if exists.trim() != "yes" {
        println!("→ создаю LXC {} ({})", a.lxc_id, a.lxc_name);
        let template = match &a.template {
            Some(t) => t.clone(),
            None => {
                let out = remote.must(
                    "pveam available 2>/dev/null | awk '/system +debian-13-standard/ {print $2; exit}'",
                )?;
                let t = out.trim().to_string();
                if t.is_empty() {
                    let out = remote.must(
                        "pveam available 2>/dev/null | awk '/system +debian-12-standard/ {print $2; exit}'",
                    )?;
                    out.trim().to_string()
                } else {
                    t
                }
            }
        };
        if template.is_empty() {
            bail!("не найден шаблон ОС (debian-13/debian-12)");
        }
        println!("  шаблон: {template}");
        remote.must("pveam update >/dev/null 2>&1 || true")?;
        let has = remote.run(&format!("pvesm list local --content vztmpl | grep -c '{template}'"))?.1;
        if !has.trim().starts_with('1') {
            remote.must(&format!("pveam download local {template}"))?;
        }
        for storage in ["local-lvm", "local", "local-zfs"] {
            let cmd = format!(
                "pct create {id} {tpl} --hostname {name} --cores {cores} --memory {mem} --swap 512 \
                 --rootfs {storage}:{disk} --ostype debian --unprivileged 1 \
                 --net0 name=eth0,bridge={bridge},ip={ip}/24,gw={gw} --onboot 1",
                id = a.lxc_id,
                tpl = template,
                name = a.lxc_name,
                cores = a.cores,
                mem = a.mem_mb,
                storage = storage,
                disk = a.disk_gb,
                bridge = a.bridge,
                ip = a.lxc_ip,
                gw = a.gateway,
            );
            let (code, _) = remote.run(&cmd)?;
            if code == 0 {
                println!("  rootfs: {storage}");
                break;
            }
        }
        let alive = remote.run(&format!("pct status {}", a.lxc_id))?.1;
        if !alive.contains("running") {
            remote.must(&format!("pct start {}", a.lxc_id))?;
        }
    } else {
        println!("→ LXC {} уже есть", a.lxc_id);
        let _ = remote.run(&format!("pct start {}", a.lxc_id));
    }

    // база + таймзона + сеть
    println!("→ база ОС, таймзона {}, сеть", a.tz);
    let _ = remote.must(&format!(
        "pct exec {id} -- bash -c 'apt-get update -qq && apt-get install -y -qq ca-certificates tzdata curl'",
        id = a.lxc_id
    ))?;
    remote.must(&format!(
        "pct exec {id} -- bash -c 'ln -sf /usr/share/zoneinfo/{tz} /etc/localtime && echo {tz} > /etc/timezone && date'",
        id = a.lxc_id,
        tz = a.tz
    ))?;
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        let (code, _) = remote.run(&format!(
            "pct exec {} -- getent hosts deb.debian.org >/dev/null 2>&1",
            a.lxc_id
        ))?;
        if code == 0 || Instant::now() > deadline {
            break;
        }
        std::thread::sleep(Duration::from_secs(5));
    }
    Ok(())
}
