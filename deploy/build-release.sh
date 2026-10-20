#!/usr/bin/env bash
# Сборка релиза в Docker-контейнере debian:bookworm.
#
# Зачем: локальная машина — Void Linux (glibc 2.41), её бинарники НЕ запустятся
# на типичном Debian/Ubuntu (glibc 2.36/2.35). Контейнер даёт бинарник,
# совместимый с Debian 12+ (PVE 8/9 обычно на Debian 12).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="debian:bookworm"
RUST_VERSION="1.99.0"
OUT="target/deploy"

echo "==> Сборка релиза в $IMAGE (rust $RUST_VERSION)"
docker run --rm \
  -v "$PWD":/src \
  -v "$PWD/target":/src/target \
  -w /src \
  -e CARGO_HOME=/tmp/cargo \
  -e RUSTUP_HOME=/tmp/rustup \
  "$IMAGE" bash -c "
    set -e
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends build-essential pkg-config ca-certificates curl git libssl-dev
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain $RUST_VERSION --profile minimal
    . /tmp/cargo/env
    cargo build --release --workspace --bin furinya
  "

mkdir -p "$OUT"
cp target/release/furinya "$OUT/furinya"
chmod +x "$OUT/furinya"
echo "==> Готово: $OUT/furinya ($(du -h "$OUT/furinya" | cut -f1))"
file "$OUT/furinya" 2>/dev/null || true
