#!/usr/bin/env python3
"""Прямой тест sd-server (stable-diffusion.cpp) без участия бота.

Гоняет набор сцен через /sdapi/v1/txt2img и складывает PNG в data/sd_tests/.
Нужен, чтобы понять: модель умеет гетерохромию/ракурсы/действия или проблема
в контейнере/настройках.

Запуск:
    python3 scripts/sd_probe.py                 # все тесты (кроме NSFW)
    python3 scripts/sd_probe.py --nsfw          # + NSFW-тест
    python3 scripts/sd_probe.py --only front,above
    python3 scripts/sd_probe.py --url http://192.168.1.172:8082/

Только stdlib, никаких зависимостей.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://192.168.1.172:8082/"
LORAS = [
    {"path": "furina.safetensors", "multiplier": 0.75},
    {"path": "Hyper-SD15-8steps-lora.safetensors", "multiplier": 0.85},
    {"path": "heterochromia-eyes.safetensors", "multiplier": 0.9},
]
NEGATIVE = (
    "worst quality, low quality, jpeg artifacts, watermark, text, signature, "
    "blurry, deformed, bad anatomy, extra fingers, extra limbs, mutated, ugly, "
    "both eyes same color, matching eyes, symmetric eyes, cross-eyed, extra eyes, "
    "dull eyes, large breasts, huge breasts, thick thighs, tall, same pose, "
    "same background, cropped head"
)

# Канон читается из репозитория, чтобы тест всегда соответствовал боту.
ROOT = Path(__file__).resolve().parent.parent
BODY_TAGS = (ROOT / "prompts/character_body.md").read_text(encoding="utf-8").strip()
CANON_TAGS = (ROOT / "prompts/character_appearance.md").read_text(encoding="utf-8").strip()
CANON = f"{BODY_TAGS}, {CANON_TAGS}"

# id, описание, сцена, доп. LoRA (None = стандартные)
TESTS: list[dict] = [
    {
        "id": "hetero_front",
        "why": "гетерохромия крупным планом, фронтальный портрет",
        "scene": "selfie, close-up portrait, front view, looking at viewer, "
                 "(heterochromia:1.3), purple eye, blue eye, soft smile, indoor lighting",
    },
    {
        "id": "hetero_nolora",
        "why": "то же без LoRA heterochromia — сравнить, даёт ли LoRA эффект",
        "scene": "selfie, close-up portrait, front view, looking at viewer, "
                 "(heterochromia:1.3), purple eye, blue eye, soft smile, indoor lighting",
        "loras": LORAS[:2],
    },
    {
        "id": "angle_above",
        "why": "ракурс сверху — проверка, слушается ли камера",
        "scene": "selfie, from above, looking up at viewer, sitting on a bed, "
                 "soft morning light, playful smile",
    },
    {
        "id": "angle_below",
        "why": "ракурс снизу",
        "scene": "selfie, from below, looking down at viewer, standing in a doorway, "
                 "dramatic evening light, confident smirk",
    },
    {
        "id": "angle_side",
        "why": "профиль/сбоку",
        "scene": "profile view, side view, sitting by a window, looking outside, "
                 "rainy evening, melancholic mood, blue tones",
    },
    {
        "id": "sit_windowsill",
        "why": "действие: сидит на подоконнике с кружкой",
        "scene": "selfie, sitting on a windowsill at dusk, holding a warm mug, "
                 "red knitted sweater, soft smile, golden sunset light, cozy room",
    },
    {
        "id": "lying_read",
        "why": "действие: лежит на кровати с книгой",
        "scene": "lying on a bed, reading a book, white oversized shirt, "
                 "morning sunlight through curtains, relaxed expression, top-down view",
    },
    {
        "id": "rooftop_wind",
        "why": "действие+место: стоит на крыше, ветер",
        "scene": "selfie, standing on a rooftop at sunset, city skyline behind, "
                 "wind blowing hair, black jacket, arms behind back, cinematic lighting",
    },
    {
        "id": "dance_neon",
        "why": "динамика: танец в неоне",
        "scene": "dancing, full body, neon night club, purple and cyan lighting, "
                 "black crop top, dynamic pose, motion energy",
    },
    {
        "id": "jump_joy",
        "why": "моушен: прыжок в воздухе, руки вверх",
        "scene": "jumping in the air, arms raised, joyful laugh, park in spring, "
                 "cherry blossoms, dynamic motion, full body, from below, action shot",
    },
    {
        "id": "run_beach",
        "why": "моушен: бежит по пляжу, волосы и вода в движении",
        "scene": "running along the beach, hair flowing in the wind, splashing water, "
                 "summer sunset, dynamic action shot, full body, side view, motion blur",
    },
    {
        "id": "stretch_morning",
        "why": "моушен: потягивается в постели",
        "scene": "stretching in bed, arms above head, arched back, yawning, messy hair, "
                 "morning sunlight, top-down view, dynamic pose",
    },
    {
        "id": "dance_spin",
        "why": "моушен: кружится в танце, юбка развевается",
        "scene": "spinning dance, skirt flaring, motion blur, stage lights, "
                 "dynamic pose, full body, low angle, energetic",
    },
    {
        "id": "hetero_strong",
        "why": "усиленная гетерохромия: (1.5) + one purple eye, one blue eye",
        "scene": "selfie, close-up portrait, front view, looking at viewer, "
                 "(heterochromia:1.5), heterochromatic eyes, one purple eye, one blue eye, "
                 "asymmetric eyes, intense gaze, indoor lighting",
    },
    {
        "id": "nsfw_bed",
        "why": "NSFW: проверка отсутствия цензуры",
        "scene": "lying on a bed, nsfw, nude, seductive pose, looking at viewer, "
                 "warm bedroom lighting, blush",
        "nsfw": True,
    },
]


def post_json(url: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/sdapi/v1/txt2img",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_server(url: str) -> None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/sdapi/v1/loras", timeout=10) as r:
            loras = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as e:
        sys.exit(f"sd-server недоступен по {url}: {e}")
    print(f"sd-server OK, LoRA: {', '.join(x.get('name', '?') for x in loras)}")


def run_test(url: str, out_dir: Path, test: dict, nsfw: bool) -> bool:
    prompt = f"{CANON}, {test['scene']}"
    if nsfw and "nsfw" not in prompt.lower():
        prompt = f"nsfw, nude, {prompt}"
    payload = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE,
        "lora": test.get("loras", LORAS),
        "steps": 8,
        "cfg_scale": 1.0,
        "width": 512,
        "height": 640,
        "sampler_name": "euler_a",
        "clip_skip": 2,
        "seed": -1,
    }
    print(f"\n── {test['id']}: {test['why']}")
    print(f"   prompt: {prompt}")
    t0 = time.time()
    try:
        resp = post_json(url, payload)
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"   ✗ ОШИБКА: {e}")
        return False
    dt = time.time() - t0
    images = resp.get("images") or []
    if not images:
        print("   ✗ пустой images[]")
        return False
    raw = base64.b64decode(images[0])
    dst = out_dir / f"{test['id']}.png"
    dst.write_bytes(raw)
    print(f"   ✓ {dst} ({len(raw) // 1024} КБ, {dt:.0f} c)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--nsfw", action="store_true", help="включить NSFW-тест")
    ap.add_argument("--only", default="", help="список id через запятую")
    ap.add_argument("--out", default=str(ROOT / "data/sd_tests"))
    args = ap.parse_args()

    check_server(args.url)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = {x.strip() for x in args.only.split(",") if x.strip()}
    tests = [
        t for t in TESTS
        if (not selected or t["id"] in selected)
        and (args.nsfw or not t.get("nsfw"))
    ]
    if not tests:
        sys.exit("нет тестов под указанные фильтры")

    ok = 0
    for t in tests:
        if run_test(args.url, out_dir, t, args.nsfw):
            ok += 1
    print(f"\nГотово: {ok}/{len(tests)} → {out_dir}")


if __name__ == "__main__":
    main()
