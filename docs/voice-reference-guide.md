# Референс-аудио голоса Фурины (`furina_ref.wav`)

> Рецепт "как быстро получить чистый референс из русской озвучки Фурины".
> Файл нужен для: (1) создания голосовой модели в **Fish Audio** (основной TTS,
> поддерживает русский), (2) запасного TTS **Voxtral** (Mistral).

## Что делаем (TL;DR)

1. Скачиваем ролик с русской озвучкой Фурины (YouTube/TikTok) — `yt-dlp`.
2. Слушаем, находим **чистый кусок 15–20 сек**: только голос Фурины,
   без музыки, без SFX, без чужих реплик.
3. Режем через `ffmpeg` в mono 44.1kHz WAV → `furina_ref.wav`.
4. Чистим тишину и нормализуем громкость.
5. Пишем транскрипт куска (для качества клонирования).
6. Загружаем в Fish Audio → получаем `referenceId` → вставляем в `config.toml`.

---

## Шаг 1. Найти и скачать ролик

Проверенные источники русской озвучки Фурины (фандаб нижних синематиков/квестов
Genshin 4.x, где её МНОГО):

- Поиск по YouTube/TikTok: `Фурина синематик русская озвучка`, 
  `Genshin Impact 4.2 аркон финал озвучка русский`, `Суд над Фуриной озвучка`.
- Лучшие сцены (много чистой речи Фурины, близко к микрофону):
  - Суд над Фуриной (конец 4.2) — её защитная речь, эмоциональные пики;
  - Монолог про 500 лет игры в роль («Диайка/спектакль»);
  - Диалоги с Невиллетом в пост-квесте 4.2 (спокойный темп — хорошо для базы).

Скачивание (нужен [yt-dlp](https://github.com/yt-dlp/yt-dlp)):

```bash
# YouTube: только аудио, wav
yt-dlp -x --audio-format wav -o "ref_raw.%(ext)s" "https://www.youtube.com/watch?v=XXXX"

# TikTok (то же самое, yt-dlp понимает tiktok-ссылки как есть)
yt-dlp -x --audio-format wav -o "ref_raw.%(ext)s" "https://www.tiktok.com/@user/video/XXXX"
```

## Шаг 2. Найти кусок (глаза/уши важнее автоматизации)

Открой `ref_raw.wav` в аудиоредакторе (Audacity / even `ffplay ref_raw.wav`) 
и ищи сегмент по критериям:

| Критерий | Почему |
|---|---|
| **Только Фурина**, ни слова никого другого | Клонирование = усреднение всех голосов в клипе |
| **Без музыки/SFX/гулкого зала** | Музыка «прилипает» к голосу и портит тембр |
| **15–20 секунд** | Fish Audio: 10–30 с на референс; Voxtral: 3–25 с (дольше — режется) |
| **Разная интонация** (падение/повышение, пауза, восклицание) | Иначе клон будет монотонным |
| Стабильная громкость | Без резких пиков/провалов |

💡 Совет: набери 2–3 кандидата по 15–20 с — выберешь лучший после проб клонов.

## Шаг 3. Резка

```bash
# Замени тайминги на свои: -ss старт, -to конец (с запасом ±1 сек, лишнее режем ниже)
ffmpeg -ss 00:01:23.500 -to 00:01:41.000 -i ref_raw.wav \
       -vn -ac 1 -ar 44100 ref_cut.wav
```

## Шаг 4. Чистка: тишина по краям + громкость

```bash
ffmpeg -i ref_cut.wav \
       -af "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.2, \
            areverse, silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.2, areverse, \
            loudnorm=I=-20:TP=-2:LRA=7" \
       furina_ref.wav
```

(сначала срезаем тишину в начале, потом `areverse+silenceremove+areverse` — в конце,
затем loudnorm выравнивает громкость под −20 LUFS).

Проверка:

```bash
ffprobe -v error -show_entries format=duration -of default=nw=1 furina_ref.wav
# должно быть 15–20 секунд
ffplay furina_ref.wav   # прослушать финальный вариант
```

## Шаг 5. Транскрипт (текст куска)

Сервисы клонирования сильно выигрывают от текста-транскрипта референса.
Свой кусок ты знаешь на слух (это сцены, которые ты смотрел), но можно
прогнать через тот же Whisper на Groq:

```bash
curl https://api.groq.com/openai/v1/audio/transcriptions \
     -H "Authorization: Bearer $GROQ_API_KEY" \
     -F model=whisper-large-v3-turbo -F language=ru -F file=@furina_ref.wav
```

Сохрани полученный текст рядом (`furina_ref.txt`) — пригодится и Fish, и Voxtral.

## Шаг 6A. Fish Audio (основной путь — работает с русским)

1. https://fish.audio → логин → **Voices / Create Model**.
2. Загрузи `furina_ref.wav` (+ текст транскрипта, + 1–3 мин ДРУГИХ чистых
   кусков, если перезаписываешь модель с нуля — больше данных = стабильнее клон).
3. После публикации скопируй **Model ID (UUID)**.
4. Вставь в `config.toml`:

```toml
[capabilities.record_voice.fish]
key = "FISH_API_KEY"
referenceId = "<UUID модели>"
```

Готово: tool `send_voice_message` в TG-режиме озвучит голосом Фурины.

> ⚠️ Правила Fish Audio: клонировать чужой голос без разрешения нельзя
> (ToS). Это фан-проект/личное использование — коммерческое применение
> (как и право на голос актрисы дубляга) запрещено.

## Шаг 6B. Voxtral TTS (запасной путь — русского ОФИЦИАЛЬНО нет)

Voxtral (Mistral) поддерживает en, fr, de, es, nl, pt, it, hi, ar. Русского нет,
но клон с русским reference частично работает (качество может плавать).

Вариант «сохранённый голос» (создаётся один раз):

```bash
MISTRAL_API_KEY="..." 
AUDIO_B64=$(base64 -w0 furina_ref.wav)

curl https://api.mistral.ai/v1/audio/voices \
     -H "Authorization: Bearer $MISTRAL_API_KEY" \
     -F name="furinya" \
     -F sample_audio="@furina_ref.wav" \
     -F languages[]="ru"
```

(если `languages[]=ru` отвергнут — создай без языка; ответ вернёт `id` — вставь
его в `config.toml` → `[capabilities.record_voice.voxtral] voiceId`).
Если multipart-вариант не примется — то же самое через официальный Python SDK:

```python
from mistralai import Mistral
import base64, pathlib
client = Mistral(api_key="...")
audio = base64.b64encode(pathlib.Path("furina_ref.wav").read_bytes()).decode()
voice = client.audio.voices.create(
    name="furinya", sample_audio=audio, sample_filename="furina_ref.wav", languages=["ru"])
print(voice.id)  # -> voiceId в config.toml
```

Файл `furina_ref.wav` должен лежать **рядом с бинарником бота** 
(путь задан в `config.toml: refAudioPath = "furina_ref.wav"`).

---

## Итоговое качество: критерии приёмки

Прослушай 3 пробных генерации на разных репликах:

- [ ] Голос узнаётся как Фурина (тембр, манера, «театральность»)
- [ ] Нет шипящих артефактов и «стекла»
- [ ] Темп речи не скомкан
- [ ] Эмоции передаются (не плоская озвучка)

Если клон плоский → возьми кусок с более выразительной интонацией.
Если шумный → ищи другой кусок (или чисти инструментом типа `demucs` /
`RVB` / Adobe Podcast Enhance, но это уже вторичная помощь).

---

# Бесплатные TTS-варианты (состояние на октябрь 2026)

## 1. Fish Audio — `s2.1-pro-free` — ПОДТВЕРЖДЕНО LIVE (октябрь 2026)

Проверено живым прогоном `cargo run -p furinya-audio --example smoke_fish`:
RU-фраза синтезирована бесплатно (104 КБ MP3) и распознана Groq слово в слово.

**КРИТИЧЕСКИЙ нюанс обмена (без него 402 Payment Required):** строку модели
Fish API требует как HTTP-ЗАГОЛОВОК, а не только JSON-поле:

```
curl -X POST https://api.fish.audio/v1/tts \
  -H "Authorization: Bearer $FISH_API_KEY" \
  -H "Content-Type: application/json" \
  -H "model: s2.1-pro-free" \
  -d '{"text":"Привет","reference_id":"<UUID клона>","format":"mp3"}'
```

JSON-поле "model" само по себе НЕ спасает — API откатится на платный
s2.1-pro и вернёт 402. В коде synthesize_fish_voice заголовок уже ставится.
В боте: `[capabilities.record_voice.fish] model = "s2.1-pro-free"`
(уже стоит) + ключ (env FISH_API_KEY) + referenceId клона Фурины.

## 2. XTTS-v2 локально — бесплатно навсегда, клон у тебя на GPU

Open-source XTTS-v2 (17 языков, русский поддержан, клон с 6–11 секунд).
У тебя GPU есть (бил local Gemma). Поднимаешь OpenAI-совместимый сервер:

```bash
# вариант A: matatonic/openedai-speech (docker)
git clone https://github.com/matatonic/openedai-speech && cd openedai-speech
docker compose -f docker-compose.cpu.yml up -d   # или .gpu.yml
# вариант B: daswer123/xtts-api-server (python)
pip install xtts-api-server && python -m xtts_api_server -d cuda:0 --deepspeed
```

Референс для XTTS — тот же `furina_ref.wav`, НО требования другие:
**mono 22050 (24000) Hz, 16-bit, 7–11 секунд**:

```bash
ffmpeg -ss 00:01:23.5 -to 00:01:34.0 -i ref_raw.wav -vn -ac 1 -ar 22050 ref_cut22.wav
ffmpeg -i ref_cut22.wav -af "silenceremove=start_periods=1:start_threshold=-50dB, areverse, silenceremove=start_periods=1:start_threshold=-50dB, areverse, loudnorm=I=-20:TP=-2:LRA=7" furina_ref_xtts.wav
```

Кладёшь файл в `voices/` и маппишь имя `furinya` в
`config/voice_to_speaker.yaml` (по README проекта). В `config.toml`
(заготовка уже есть — раскомментируй и переключи backend):

```toml
record_voice.backend = "openai"
[capabilities.record_voice.openai]
url = "http://localhost:8000/v1/"
key = "sk-111111111"
model = "tts-1-hd"
voice = "furinya"
```

Плюсы: бесплатно навсегда, без квот, оффлайн. Минусы: качество ниже Fish
S2.1; ~4GB VRAM; Coqui CPML-лицензия (некоммерческое — для Furinya ок).

## 3. Прочие ограниченно-бесплатные (для сравнения)

| Сервис | Бесплатно | Клон | Русский | Вердикт |
|---|---|---|---|---|
| **Gradium** | 45k кред/мес + 5 клонов, без карты | да | **нет** (en/fr/de/es/pt) | отпадает |
| **Gandr** | 50k символов разово | да (5–10с) | да (beta) | годен для пробного теста клона |
| **SiliconFlow** | триал-кредиты | да | fish-speech-1.5: ru | дешёвый платный после триала |
| **MiniMax Speech** | ~10k кред/мес (менялись) | да | 32 языка | нестабильный фритир |
| **ElevenLabs** | 10k символов/мес | клон только платно | да | не Фурина |
| **Replicavox** | 15k символов/мес | да | не проверен | малый лимит |
| **Google Cloud TTS** | 4M символов/мес WaveNet | нет | ru-RU-Wavenet | базовый запас, не Фурина |

Рекомендация: сначала Fish `s2.1-pro-free` (10 минут), параллельно перепиши
`furina_ref.wav` под XTTS (22050Hz mono, 7–11с) и подними независимый
бэкенд — чтобы голос не зависел от чужого fair-use.
