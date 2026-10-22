# Phase 2 — Media primitives: from a Script to a video file

> Read at your own pace. Ask about anything, in any order. Parts 3–5 (ffmpeg, assembly, tests)
> are completed from the build session's notes once the build is done.

---

## 0. The big picture

Phase 1 ended with a validated `Script` (title, hook, scenes, CTA). Phase 2 turns it into an mp4
**without any agents yet**: plain functions, called in order.

```
Script (Gemini, Phase 1)
  │
  ├─► images.py   one picture per scene          (Step 2.1)  placeholder now, FLUX later
  ├─► tts.py      one voice file per scene        (Step 2.2)  + word timings → subtitles.srt
  │
  ├─► ffmpeg.py   picture → moving clip           (Step 2.3)  zoom / pan ("Ken Burns")
  └─► assembly    clips + voices + subtitles      (Step 2.4)  + music → runs/<id>/final.mp4
```

Every run lives in its own folder, so nothing ever overwrites an earlier run:

```
runs/2026-10-21_160447/
  script.json          the Script the LLM wrote
  scenes/scene_01.png  one image per scene
  audio/scene_01.mp3   one voice file per scene
  subtitles.srt        word-timed subtitles for the whole video
  final.mp4            (Step 2.4)
```

**Where the money goes** (one 30-second video, 6 scenes):

| Part | Tool | Cost |
|---|---|---|
| Script | Gemini free tier | $0 (list price ~$0.002) |
| Images | placeholders now; FLUX later | $0 now; ~$0.003 each later |
| Voice | edge-tts | $0 |
| Motion, subtitles, music | ffmpeg on your PC | $0 |
| Hero clip (later, 1 per video) | fal image-to-video | ~$0.25–0.60 |

So the whole of Phase 2 is built at **$0**.

---

## 1. Step 2.1 — Images (`src/shortsmith/media/images.py`)

### 1.1 Concept: background-job APIs
Generating an image takes seconds; a video can take minutes. That's too long to keep one web
request open, so services like fal use a **queue**:

1. **Submit** the job → you get a request id (a restaurant buzzer).
2. **Wait** until it's done. Three ways to wait:
   - **polling**: ask "done yet?" again and again;
   - **subscribe**: a library does the polling for you;
   - **webhook**: the service calls *your* server when it's done (needs a server).
3. **Result**: JSON with a **URL** of the file → **download** it.

`fal_client.subscribe(...)` shows `interval=0.1` in its signature: it asks the queue every
0.1 seconds. "Subscribe" is polling wrapped in a convenient function.

### 1.2 Concept: mock mode
The pipeline must be buildable and debuggable without spending money. So `generate_image()` has
**two implementations behind one signature**:

- **mock** → `_placeholder()`: a coloured 1080×1920 card with the scene label and the prompt;
- **real** → `_flux()`: FLUX schnell on fal.

The rest of the pipeline only ever calls `generate_image()`, so switching to real images later
changes **one setting**, not code. This "same interface, swappable implementation" idea comes back
in Phase 3 (tools) and Phase 4 (MCP).

### 1.3 The code, function by function

**`mock_media()`**
```python
return os.environ.get("MOCK_MEDIA", "1") != "0"
```
- `os.environ.get("MOCK_MEDIA", "1")`: read the setting; if it's missing, pretend it's `"1"`.
- `!= "0"`: only an explicit `"0"` switches mock **off**.
- **Fail-safe default.** When one side of a switch costs money, every mistake must land on the
  free side: a missing setting, a typo (`MOCK_MEDIA=no`), a fresh `.env` → all free.
  With `== "1"` instead, all of those would spend money.

**`generate_image(prompt, out_path, label="", step="producer", ledger_path=...)`**
- Creates the folder (`out_path.parent.mkdir(parents=True, exist_ok=True)`), then picks mock or
  real. `step="producer"` is who will call it in Phase 6 (the Producer agent); it ends up in the
  ledger.

**`_placeholder(prompt, out_path, label)`**
```python
seed = zlib.crc32(prompt.encode("utf-8"))
colour = tuple(60 + (seed >> shift) % 140 for shift in (16, 8, 0))
```
- `zlib.crc32` turns the prompt into a number (a checksum). **Same prompt → same number → same
  colour**, so re-running gives identical pictures (tested).
- `seed >> 16`, `>> 8`, `>> 0` take three different slices of that number for red, green, blue.
- `60 + … % 140` keeps each channel between 60 and 199: **mid-tones**, never too dark or too
  light, so white text is always readable.
- `textwrap.wrap(prompt, width=30)` breaks the prompt into lines;
  `ImageFont.load_default(size=52)` is Pillow's built-in scalable font (no font file needed on any
  machine); `multiline_text` draws it.

**`_flux(prompt, out_path, step, ledger_path)`** — the order follows the money:
1. `price = ledger.MEDIA_PRICES[FLUX_MODEL]` **first**: an unpriced model raises *before* paying
   (same "fail before you pay" rule as `chat()` in Phase 1).
2. `fal_client.subscribe(...)` submits and waits. Arguments: `image_size="portrait_16_9"`
   (576×1024; FLUX is billed by megapixels, and ffmpeg scales it up later), `num_images=1`,
   `output_format="png"`.
3. `ledger.record(...)` **right after generation**: fal charges when the image is made, so a
   failed download must not hide a paid image from the budget.
4. `_download(url, out_path)` last.

**`_download(url, out_path)`** is a one-line function on purpose: it's a **seam**, a spot where
tests can swap in a fake (`monkeypatch.setattr(images, "_download", ...)`), so the real path is
tested without internet or money.

**`ledger.py` additions**
- `MEDIA_PRICES = {"fal-ai/flux/schnell": 0.003}`: media is priced **per item**, not per token.
  Images are recorded with 0 tokens + their item price, so one ledger covers LLM and media.
- `DEFAULT_PATH = Path("runs/ledger.jsonl")`: one place defines where the ledger lives; `llm.py`
  and `images.py` both use it.

**The demo (`__main__`)**: topic → `write_script()` → one image per scene, saved in a timestamped
run folder with `script.json`. The `write_script` import sits *inside* `__main__`, so the images
module itself doesn't depend on the LLM.

### 1.4 The tests (`tests/test_images.py`)
| Test | What it proves |
|---|---|
| `test_mock_is_the_default_and_only_0_turns_it_off` | the fail-safe default |
| `test_placeholder_is_a_vertical_full_hd_png` | real Pillow output: 1080×1920, PNG |
| `test_same_prompt_gives_the_same_placeholder_colour` | crc32 → stable colours |
| `test_flux_path_queues_records_and_downloads` | fake `subscribe` + fake `_download`: right model and size sent, file written, $0.003 in the ledger |

### 1.5 Image + camera motion vs AI video clip (your questions)
- **AI image + camera motion**: one still picture; ffmpeg moves a "camera" over it (zoom or
  slide). **Nothing inside the picture moves**, only the framing. Called the *Ken Burns effect*.
  Cost: one picture (~$0.003).
- **AI video clip**: an AI model draws **every frame** (~120 frames for 5 seconds), so steam
  rises and leaves fall. Usually "image-to-video": the scene's picture is the first frame.
  Cost: priced per second.
- **Hero clip**: the one scene per video (usually the hook) that gets a real AI clip. That's what
  `ai_clip` in `models.Motion` is for. Always approval-gated, never made while `MOCK_MEDIA=1`.
- **Real prices (checked 2026-10-21):** a 5-second clip costs ~$0.25–0.60 on fal (Wan 2.5 480p
  and Veo 3.1 Lite from ~$0.05/s, Kling ~$0.07–0.11/s, premium models $0.30–0.75/s). The plan's
  old $0.10–0.25 was too optimistic.
- **$5 is a prepaid balance, not a per-video price:** images only ≈ 270 videos; your target
  (6 images + 1 hero clip, ~$0.27–0.62) ≈ 8–18 videos; every scene as a clip ≈ 1–3 videos.

---

## 2. Step 2.2 — Voice and subtitles (`src/shortsmith/media/tts.py`)

### 2.1 Concept: text-to-speech with edge-tts
- **edge-tts** talks to the voice service behind Microsoft Edge's "Read aloud": free, no key,
  good voices, including Uzbek. **Risk:** it's an *unofficial* use of that service, so Microsoft
  could rate-limit or block it. That's why it sits behind one function (`synthesize`), like images.
- Voices used (`VOICES`): `en-US-AndrewNeural`, `uz-UZ-MadinaNeural`, `ru-RU-SvetlanaNeural`.
  (Also available: `uz-UZ-SardorNeural`, `en-US-AvaNeural`, `ru-RU-DmitryNeural`, …)
- **Word timings**: while speaking, the service sends a `WordBoundary` event for every word:
  ```
  {'type': 'WordBoundary', 'offset': 1250000, 'duration': 5375000, 'text': 'Toshkentda'}
  ```
  Times are in **"ticks" of 100 nanoseconds**: 10,000,000 ticks = 1 second, so
  1,250,000 → 0.125 s. The library's *default* is sentence timings; we must ask for
  `boundary="WordBoundary"`. We verified that Uzbek voices send word timings too.

### 2.2 Concept: `async` / `await` (new Python idea)
- Network work is mostly **waiting**: send a request, wait for bytes to arrive.
- A normal function **blocks** while it waits. An `async def` function is a **coroutine**: at each
  `await` it says "I'm waiting; let something else run meanwhile".
- The **event loop** is the scheduler that runs coroutines and wakes them up when their data has
  arrived.
- `async for chunk in communicate.stream()` = "for every message that arrives from the network",
  waiting between them without blocking.
- `asyncio.run(coroutine)` starts an event loop, runs the coroutine to the end and returns its
  result. It's the **bridge** from normal code into async code; everything outside `_speak` stays
  ordinary Python.
- Why it matters later: with async you can run several network calls **at the same time**
  (`asyncio.gather`), e.g. voicing all 6 scenes in parallel. We don't yet, to be gentle with a free
  unofficial service.

### 2.3 Concept: the audio decides the timing
In Phase 1 the LLM *guessed* each scene's `duration_s`. Measured reality:

| | planned by the LLM | real speech |
|---|---|---|
| English (6 scenes) | 28.0 s | 17.4 s |
| Uzbek (6 scenes) | 29.0 s | 18.3 s |

The guesses were ~40% too long, and the languages differ: English ~2.6 words per second, Uzbek
~1.6–1.9 (longer, agglutinative words). An LLM can't estimate speech length, so **each scene now
lasts exactly as long as its real audio**, measured by ffprobe. Subtitles then come from the same
audio's word timings, which is why they're always in sync. (The build session changes the writer
prompt to target word counts per language.)

### 2.4 Concept: the `.srt` subtitle format
```
1
00:00:00,138 --> 00:00:01,612
Kuzda qayerda kofe ichishni

2
00:00:01,612 --> 00:00:02,200
bilasizmi
```
Number, `start --> end`, text, blank line. Note the **comma** before the milliseconds. Every
player and ffmpeg understands it.

### 2.5 The code, function by function

**Models**: `Word(text, start_s, end_s)` and `Voiceover(audio_path, duration_s, words)`.
Pydantic again: data passed between steps is typed, never loose dicts.

**`_speak(text, voice, out_path)`** (the async part)
- `edge_tts.Communicate(text, voice, boundary="WordBoundary")`: ask for word timings.
- Opens the mp3 file in binary mode (`"wb"`); for each chunk:
  - `"audio"` → write its bytes to the file;
  - `"WordBoundary"` → convert ticks to seconds, append a `Word`.

**`audio_duration(path)`** — our first `subprocess` call
```python
subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
```
- The command is a **list of arguments**, not one string: no shell is involved, so spaces in paths
  (`D:\All codes\…`) can't break it and nothing gets misinterpreted.
- `capture_output=True` collects what ffprobe prints; `text=True` gives a `str` instead of bytes.
- `check=True`: a non-zero exit code raises an error instead of returning garbage.
- ffprobe prints just the number (e.g. `2.304`), so `float(result.stdout)` is the duration.
- Step 2.3 uses exactly this pattern with ffmpeg.

**`synthesize(text, voice, out_path)`** — with retry and backoff
```python
for attempt in range(1, ATTEMPTS + 1):
    try:
        words = asyncio.run(_speak(...)); break
    except edge_tts.exceptions.NoAudioReceived:
        if attempt == ATTEMPTS: raise
        time.sleep(2 ** (attempt - 1))
```
- **Why it exists (an error that was a lesson):** the first English run failed with
  `NoAudioReceived: No audio was received`. I probed the same voice and text 30 seconds later:
  everything worked. So it wasn't our code, the voice or the text, but a **transient** hiccup of
  the free service.
- **Retry with exponential backoff:** wait 1 s, then 2 s (`2 ** 0`, `2 ** 1`), then give up loudly
  (`raise`), because a silent scene would ruin the video. Waiting longer each time gives the
  service room to recover instead of hammering it. (Production systems often add random "jitter"
  to the waits so many clients don't retry at the same instant.)
- Only `NoAudioReceived` is retried: other errors (wrong voice name, no internet) fail immediately.

**`srt_time(seconds)`**
```python
ms = round(seconds * 1000)
hours, ms = divmod(ms, 3_600_000); minutes, ms = divmod(ms, 60_000); secs, ms = divmod(ms, 1000)
```
`divmod(a, b)` returns (quotient, remainder): peel off hours, then minutes, then seconds, like
reading a clock. `f"{hours:02}"` pads to two digits.

**`group_words(words, per_line=4)` and `to_srt(lines)`** — and the bug a test caught
- My first version grouped words 4 at a time **across the whole video**. The test with one word per
  scene produced the line `scene 1 scene 2 scene 3 scene 4`: one subtitle line spanning **four scene
  cuts**, which in a real video would hang on screen across cuts and pauses.
- Fix: `group_words()` is called **per scene**, and `to_srt()` just numbers all the lines. The test
  now asserts one line per scene (`srt.count(" --> ") == 5`).

**`shift(words, by_s)`**: a scene's word times start at 0; shifting by "where this scene starts in
the whole video" (the running total of real durations) puts them on the global timeline.

**`voice_script(script, run_dir)`** *(as changed by the build session)*: for each scene →
`synthesize()`, then it saves **`voiceovers.json`**: every scene's audio path and word timings, so a
run can be re-edited later with no new voice calls. It no longer writes the subtitles itself: the
editor owns the video's timeline (section 4), so it calls **`subtitles(voiceovers, scene_starts)`**,
which shifts and groups each scene's words exactly as described above.
- `_VOICEOVERS = TypeAdapter(list[Voiceover])`: Pydantic validation for a plain *list* of models
  (a `BaseModel` can't be a bare list), used to write and read the JSON.
- Audio paths are saved **relative to the run folder** (`audio\scene_01.mp3`). That fixed a real bug
  (error E10 in section 6).
- `load_voiceovers(run_dir)` reads it back, or returns `None` if anything is missing, which means
  "voice it again" (edge-tts is free).

**`__main__`**: `max(Path("runs").glob("*/script.json"))` picks the **newest** run (timestamp folder
names sort in time order), voices it and prints planned vs real seconds per scene.

### 2.6 The tests (`tests/test_tts.py`)
| Test | Technique |
|---|---|
| `test_srt_time_uses_a_comma_before_milliseconds` | pure function, exact strings |
| `test_to_srt_groups_words_into_numbered_blocks` | exact SRT output |
| `test_shift_moves_words_to_the_scene_start` | pure function |
| `test_speak_writes_audio_and_converts_ticks` | `FakeCommunicate` whose `stream()` is an **async generator** (`async def` + `yield`); also asserts we asked for `WordBoundary` |
| `test_synthesize_retries_when_no_audio_arrives` | fails twice, then works → 3 attempts; `time.sleep` patched so the test doesn't wait |
| `test_synthesize_gives_up_after_the_last_attempt` | always fails → the error comes out |
| `test_voice_script_writes_one_srt_with_shifted_scenes` | fake `synthesize`; checks the shift, one line per scene, and the Uzbek voice |

Running async code inside a normal test: `asyncio.run(tts._speak(...))`. No extra pytest plugin
needed.

---

## 3. Step 2.3 — Camera motion with ffmpeg 🧠 (`src/shortsmith/media/ffmpeg.py`)

### 3.1 Concept: how an ffmpeg command is built
Every ffmpeg command has the same four parts:

```
ffmpeg  [inputs: -i file ...]  [filters: -vf / -filter_complex]  [encoders: -c:v, -c:a ...]  output
```

- **Inputs** are files (or generated sources, see the music bed).
- **Filters** process the decoded frames/samples. A **filter graph** is a pipeline of filters:
  - a *simple chain* (`-vf`): filters joined by `,` run one after another:
    `scale=...,crop=...,zoompan=...`
  - a *complex graph* (`-filter_complex`): several chains joined by `;`, with **labels** in square
    brackets saying where streams come from and go to: `[0:v]` = input 0's video, `[1:a]` = input 1's
    audio, `[v]` / `[a]` = names you invent, later picked with `-map [v] -map [a]`.
- **Encoders** compress the result (`libx264` for H.264 video, `aac` for audio).
- Every option value goes through several "languages" at once: Python string → the argument list →
  ffmpeg's filter syntax (where `:` separates options, `,` separates filters and `'` quotes) →
  sometimes a library inside a filter (libass for subtitles). Escaping must be right at **every**
  layer, which is why the code avoids awkward characters altogether (decision D3, error E1).

There's an **interactive explainer** for the zoompan filter: open it from the chat link and drag the
progress slider for each motion.

### 3.2 Concept: frames are integers
At 30 fps a video can only cut on a frame boundary: 3.0 s = 90 frames, 3.01 s is *still* 90.
`frames(seconds) = max(1, round(seconds * 30))` converts every duration **once**, and every later
step (picture, voice, subtitles) reuses the same integers. Section 4 shows why that matters.

### 3.3 Concept: the zoompan filter
`zoompan` takes **one** input frame (the still image) and outputs `d` frames. For every output frame
it evaluates three **expressions**:

1. `z`: the zoom of this frame (1 = whole picture, 1.2 = the window shows 1/1.2 = 83% of it);
2. `x`, `y`: the top-left corner of the visible window, in input pixels;

then it cuts that window (`iw/zoom` wide, `ih/zoom` high) and scales it to `s=1080x1920`.
Variables inside the expressions: `on` = output frame number (0 … d−1), `iw`/`ih` = input size,
`zoom` = this frame's zoom. So **`on/(d-1)` is the progress, from 0 to 1**.

| Motion | `z` | `x` | `y` |
|---|---|---|---|
| zoom_in | `1+0.2*on/(d-1)` (1 → 1.2) | `(iw-iw/zoom)/2` (centred) | `(ih-ih/zoom)/2` |
| zoom_out | `1.2-0.2*on/(d-1)` (1.2 → 1) | centred | centred |
| pan_right | `1.2` fixed | `(iw-iw/zoom)*on/(d-1)` (left edge → right edge) | centred |
| pan_left | `1.2` fixed | `(iw-iw/zoom)*(1-on/(d-1))` (right → left) | centred |
| ai_clip | like zoom_in until real clips exist | | |

- **Why `(iw-iw/zoom)/2` centres the window:** the picture is `iw` wide, the window `iw/zoom`; the
  leftover space is `iw-iw/zoom`, and half of it on each side puts the window in the middle.
- **Why panning needs zoom 1.2:** at zoom 1 the window is the whole picture, so there's no room to
  slide. At 1.2 there's 17% of the width to travel across.
- `max(d-1, 1)` in Python: a 1-frame clip would otherwise divide by zero.

### 3.4 Concept: why upscale before zoompan
zoompan places its window in **whole input pixels**. A slow 5-second drift moves the window less than
one pixel per frame, so on a small picture the movement goes "step, step, step": visible jitter.
Upscaling to **2× the output (2160×3840)** first makes each input pixel half an output pixel, so the
steps are half as big. (4× would be smoother but slower.)
- `scale=2160:3840:force_original_aspect_ratio=increase:flags=lanczos`: grow the picture until it
  **covers** 2160×3840 (never letterbox); `lanczos` is a sharp resampling method. FLUX's 576×1024
  and the placeholder's 1080×1920 both end up 2160×3840.
- `crop=2160:3840`: if an image ever isn't 9:16, cut off the extra instead of squashing it.

### 3.5 `ken_burns_cmd(image, out, duration_s, motion)` line by line
The real command for one Uzbek scene (182 frames, pan_left):
```
ffmpeg -hide_banner -loglevel error -y -i scenes/scene_02.png
  -vf "scale=2160:3840:force_original_aspect_ratio=increase:flags=lanczos,
       crop=2160:3840,
       zoompan=z='1.2':x='(iw-iw/zoom)*(1-on/181)':y='(ih-ih/zoom)/2':d=182:s=1080x1920:fps=30"
  -frames:v 182 -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p clips/scene_02.mp4
```
- `FFMPEG = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]`: no banner, only real errors, `-y`
  overwrites old outputs without asking.
- It's a **list**: no shell ever splits it, so spaces in paths are safe.
- `-frames:v n`: a guard so the clip has exactly n frames.
- `-crf 18 -preset veryfast`: CRF is "constant quality" (lower = better, 18 ≈ visually lossless);
  the preset trades speed for file size. Near-lossless because these clips are re-encoded once more
  in the final step.
- `-pix_fmt yuv420p`: a PNG gives full-colour 4:4:4 video, which many phones and QuickTime can't
  play. 4:2:0 is what every device decodes.

### 3.6 🧠 Your exercise (whenever you like, never blocking)
- File: `exercises/phase-2/ken_burns.py`, the same signature with hints, body
  `raise NotImplementedError`.
- Check it (terminal, repo root):
  ```
  uv run pytest exercises/phase-2
  ```
  16 tests, all red now. They **evaluate your expressions in Python** for every frame (decision D12):
  zoom starts at 1.0 and ends at 1.2, the window stays centred, a pan goes edge to edge, the window
  never leaves the picture. So **any correct formula passes**, not just one exact string. Use only
  numbers, `+ - * / ( )`, `min`, `max` and the names `on`, `zoom`, `iw`, `ih`.
- Then compare with the real `ken_burns_cmd()` in `ffmpeg.py`. Ask me to review your version anytime.

---

## 4. Step 2.4 — Assembling `final.mp4` (`ffmpeg.py`, `make.py`)

### 4.1 Concept: picture and sound meet only once, cut to the same integers (decision D2)
The tempting way: make each scene a clip *with* its own audio, then glue the clips together. It drifts:
every AAC audio piece carries ~20–40 ms of encoder "priming" padding, so at each join the sound slips a
little further from the picture and can click.

What the build does instead:
- the scene clips are **silent**;
- every scene's voice is padded/cut to **exactly** `frames × 1600` audio samples. At 48,000 samples
  per second and 30 frames per second, one frame = **1600 samples**, an exact integer;
- the subtitles are shifted by the **same** frame counts.

Video, voice and subtitles share **one list of integers**, so they *can't* drift apart. The build
checked it: speech onsets detected in the final mix match the subtitle start times with a constant
~0.05 s offset (the detector's reaction time) that **doesn't grow** from scene 2 to scene 6.

### 4.2 How long each scene stays on screen (decision D1)
`scene_frames(voiceovers)`: until the scene's **last spoken word** + 0.3 s (the last scene + 1.0 s, so
the video doesn't stop on the final syllable). Why not the mp3's full length? Measured: edge-tts files
end in **0.37 s (English) and 0.93 s (Uzbek) of silence**. Using the file length would leave ~1.4 s
of dead air between Uzbek scenes, which feels sluggish in a short.
`scene_starts(counts)` uses `itertools.accumulate` (a running sum) to find where each scene starts.

### 4.3 The final command, part by part
```
ffmpeg ... -f concat -i clips.txt                 input 0: the 6 silent clips, back to back
           -i audio/scene_01.mp3 ... scene_06.mp3  inputs 1-6: one voice per scene
           -stream_loop -1 -i music.wav            input 7: music, looped if shorter than the video
  -filter_complex "<the graph below>"  -map [v] -map [a]  -frames:v 863
  -c:v libx264 -preset medium -crf 21 -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart final.mp4
```
**Video**
- `-f concat -i clips.txt`: the **concat demuxer** reads the clips one after another as if they were
  one file. It works because all clips have identical settings. `clips.txt` lists them:
  `file 'clips/scene_01.mp4'`. (Not to be confused with the `concat` **filter** below, which joins
  *decoded* audio.)
- `[0:v]subtitles=subtitles.srt:force_style='...'[v]`: **burns** the SRT into the picture with the
  libass library:
  - `PlayResX=1080,PlayResY=1920`: makes font size and margins real video pixels (by default an SRT
    gets a tiny 384×288 canvas and everything is scaled from that);
  - `FontName=Arial,FontSize=84,Bold=1`: big and readable; Uzbek `o'`/`g'` render fine;
  - colours are `&HAABBGGRR`: alpha first, then **blue-green-red** (backwards!); `00` alpha = opaque.
    White text `&H00FFFFFF`, black outline `&H00000000`;
  - `BorderStyle=1,Outline=6`: a 6-px outline (not a box);
  - `Alignment=2`: bottom centre, using the **numpad** layout (1 = bottom-left … 9 = top-right);
  - `MarginV=560`: the text's bottom edge sits 560 px above the bottom, about 70% down the frame,
    clear of the apps' buttons and captions.

**Voice**
- `[1:a]aformat=sample_rates=48000:channel_layouts=stereo`: edge-tts gives 24 kHz mono MP3;
  everything is converted to one format before joining.
- `apad=whole_len=N` then `atrim=end_sample=N`: pad with silence to *at least* N samples, then cut to
  *exactly* N (N = that scene's frames × 1600).
- `concat=n=6:v=0:a=1`: join the six pieces (0 video, 1 audio stream).
- `asplit=2[voice][key]`: two copies of the voice: one to hear, one to steer the ducking.

**Music**
- `atrim=end_sample=<total>`: cut the looped music to the video's length.
- `volume=-20dB`, `afade=t=in:d=1.5`, `afade=t=out:st=…:d=2.5`: quiet, fades in and out.
- `[bed][key]sidechaincompress=threshold=0.03:ratio=4:attack=20:release=400`: **ducking**. The music is
  compressed according to the loudness of the *other* input (the voice copy). When the voice rises
  above the threshold (≈ −30 dBFS) the music goes down (ratio 4:1) within 20 ms, and comes back up
  slowly (400 ms) in the pauses. That's the "radio presenter" effect.

**Mix and loudness**
- `amix=inputs=2:duration=first:normalize=0`: voice + ducked music. `normalize=0` stops amix from
  dividing each input by the number of inputs, which would make the voice 6 dB quieter.
- `loudnorm=I=-14:TP=-1.5:LRA=11`: **loudness normalisation** (EBU R128) to **−14 LUFS**, the level
  YouTube and TikTok play at. LUFS measures *perceived* loudness (how loud it sounds to a human), not
  just peaks. YouTube turns loud videos down but never quiet ones up, so a quiet video just sounds weak.
- `aresample=48000`: loudnorm works at 192 kHz internally, so the output is converted back.

**Output**
- `-frames:v 863`: exactly the sum of all scene frames.
- `-crf 21 -preset medium`: a smaller file than the near-lossless clips (2.6–2.8 MB for 30 s).
- `-movflags +faststart`: moves the MP4's index to the front of the file, so phones can start
  playing before the whole file has downloaded.

### 4.4 The music bed: generated, never downloaded (decision D5)
Downloading music raises licence questions and untrusted files, so the build **generates** a soft
placeholder pad with ffmpeg itself:
```
-f lavfi -i "aevalsrc='0.45*( <chord 1> + <chord 2> + <chord 3> + <chord 4> )':s=48000:d=30.067"
-af "lowpass=f=1500,aecho=0.8:0.5:90|180:0.3|0.2"
```
- `-f lavfi -i "..."`: the "input" is a filter that **creates** audio instead of reading a file.
- `aevalsrc` evaluates a formula for every sample (`t` = time in seconds). Each chord is the sum of
  4 sine waves (`sin(2*PI*f*t)`, f = the note's frequency): **Am → F → C → G**, one every 4 s.
- Each chord is multiplied by a **sin² window** 8 s long. Neighbouring windows overlap by half, and
  sin² + cos² = 1, so the windows **add up to exactly 1**: the pad never clicks or dips at a change.
- `lowpass=f=1500` removes harsh high frequencies (softer); `aecho` adds two short echoes (a "room").
- `--music data/music/your_track.mp3` uses your own track instead (that folder is gitignored, for
  licensed music).

### 4.5 `run()` and `probe()`
- `run(cmd, cwd)`: `subprocess.run(..., check=True)` like ffprobe in Step 2.2, plus
  `e.add_note(f"ffmpeg said: ...")` (Python 3.11+): when ffmpeg fails, the traceback shows ffmpeg's
  own error message instead of just "exit status 1".
- `probe(path) -> VideoInfo`: ffprobe's JSON output → a Pydantic model (size, fps, frames, duration,
  codecs). Used by the summary and the live tests.

### 4.6 Why every command runs inside the run folder (decision D3)
The `subtitles=` filter takes a **file name inside a filter graph**, where `:` separates options and
`\` and `'` are escape characters. A path like `D:\All codes\shortsmith\runs\...\subtitles.srt`
contains `:`, `\` and a space: it would need several layers of escaping. Running ffmpeg with
`cwd=run_dir`, the name is just `subtitles.srt`, and `final_cmd` refuses anything that isn't a plain
name (`re.fullmatch(r"[\w.-]+", ...)`). Normal inputs (`-i`) are ordinary arguments, so even your own
music path with spaces is fine there.

### 4.7 Fixing the duration problem from Step 2.2 (decision D9)
- `models.py`: the old rule "total `duration_s` 20–40 s" checked the LLM's *guesses*. It's replaced by
  a **word-count rule per language**, measured on the real voices (pauses included):
  `NARRATION_WORDS = {"en": (78, 84), "uz": (55, 60), "ru": (58, 64)}` (English ~2.7 words per second,
  Uzbek ~2.0; Russian is still an estimate). `WORDS_SLACK = 0.25` rejects only scripts that are far
  off; after voicing, `make` *warns* if the real video is outside 25–35 s.
- **Per scene, not only in total (error E7):** Gemini Flash-Lite wrote 34 Uzbek words when asked for
  55–60 in total. Asked for "9–10 words per scene" next to the total, it hit 56. **An LLM hits a small
  number far better than a big one.** Same lesson as the hook in Phase 1.
- `WRITER_SYSTEM_PROMPT` is now an **f-string built from `NARRATION_WORDS`**, so the prompt and the
  validator can never disagree (a test checks it). It asks for exactly 6 scenes (= 6 images).
- **The CTA is now spoken** (decision D10): the last scene's narration ends with it, like the hook
  opens scene 1. Before, the `cta` field was never voiced or shown.
- **Old runs still work (error E8):** `Script.load(path)` validates a saved `script.json` with
  `context={"saved": True}`, and the validator (`info: ValidationInfo`) skips the word-count rule for
  saved scripts but still checks the structure. A script written under an older rule must stay
  re-assemblable.

### 4.8 `make.py`: the whole pipeline in one command
```
uv run --env-file .env python -m shortsmith.make "coffee shop in Tashkent, cozy autumn vibe" --lang uz
uv run --env-file .env python -m shortsmith.make --run runs/2026-10-22_112757   (re-edit, no LLM)
uv run --env-file .env python -m shortsmith.make --run runs/... --music data/music/track.mp3
```
- `new_run()` is the **only LLM call**: write the script, save it in a new timestamped folder.
- `make_images()` skips images that already exist: **in paid mode an image is never bought twice**.
- `make_voice()` reuses `voiceovers.json` if it matches the script; otherwise it voices again.
- `ffmpeg.assemble()`: subtitles → clips → `clips.txt` → music → final command.
- `summary()`: planned / voice file / on screen / words per scene, ffprobe facts, a WARNING outside
  25–35 s, and this run's cost (ledger total after minus before).
- `@contextmanager def step(name)`: a small context manager: the code before `yield` runs when the
  `with` block starts (print the name, start the stopwatch), the code after `yield` when it ends
  (print the seconds). One line per pipeline step, e.g. `voice (uz-UZ-MadinaNeural)... 15.2 s`.
- `if (args.topic is None) == (args.run is None)`: exactly one of the two must be given.

### 4.9 Results (real numbers, $0)
| | English | Uzbek |
|---|---|---|
| Topic | "coffee shop in Tashkent, cozy autumn vibe" | "Samarqand, Registon maydoni kechqurun" |
| File | `runs/2026-10-22_112855/final.mp4` | `runs/2026-10-22_112757/final.mp4` |
| Length | **30.1 s** (902 frames) | **28.8 s** (863 frames) |
| Words | 75 | 56 |
| Size | 2.8 MB, 1080×1920, 30 fps, H.264 + AAC | 2.6 MB, same |
| Loudness | −14.3 LUFS | −14.7 LUFS |
| Time to make | 79 s (voice 47 s, ffmpeg 28 s) | 47 s (voice 15 s, ffmpeg 27 s) |
| Cost (list price) | $0.0016 | $0.0019 |

With word targets, plan and reality now agree within ~1 s (Step 2.2: planned ~29 s, real ~18 s).

---

## 5. Step 2.5 — Testing ffmpeg commands

- **Pure command builders** (`zoompan`, `ken_burns_cmd`, `music_bed_cmd`, `final_cmd`, `scene_frames`,
  `scene_starts`) return lists and strings, so the tests check them **without running ffmpeg**: fast and
  exact.
- **`assemble()` with a faked `run()`**: the test records the commands instead of executing them, then
  checks their order, the files written (`subtitles.srt`, `clips.txt`) and the frame counts.
- **zoompan tests evaluate the expressions** (decision D12): `eval()` runs them with `on`, `zoom`, `iw`,
  `ih` bound, for every frame, and checks where the camera goes. Any correct formula passes, which is
  exactly what your exercise needs.
- **Only two live tests** (decision D13): render a 1-second clip from a 576×1024 and from a 1080×1920
  image, then `probe()` checks 1080×1920, 30 frames, H.264/yuv420p. Skipped if ffmpeg isn't installed.
- **76 tests in total**: test_ffmpeg 31, test_tts 10, test_models 9, test_ledger 8, test_llm 7,
  test_make 7, test_images 4. Your exercise has its own 16 in `exercises/phase-2/`.
- `[tool.pytest.ini_options] testpaths = ["tests"]` in `pyproject.toml`: a plain `uv run pytest` runs only
  the real tests, so your unfinished exercise never turns the suite red.

---

## 6. Errors that happened, and what they teach
| # | What happened | How it was found | Fix and lesson |
|---|---|---|---|
| E1 | A shell heredoc turned `\\n` in Python f-strings into real line breaks | reading the file back right after editing | edit files with a proper editor tool; **escaping changes at every layer** |
| E2 | The music bed was inaudible (−25.6 LUFS before −20 dB) | **measured** with ffmpeg's `ebur128`, not by ear | louder bed (−14.7 LUFS) so one `volume=-20dB` fits any track |
| E3 | Whole video too quiet (−20.9 LUFS) | measured | `loudnorm` to −14 LUFS: platforms never turn quiet videos up |
| E4 | `ebur128` printed hundreds of lines | it flooded the terminal | filter the output for the summary lines |
| E5 | Two of the build's own tests were wrong (a 4-item slice compared to 3 items) | the test run | fix the test, not the code: **tests can be wrong too** |
| E6 | First English video 22.7 s instead of ~30 | the summary's WARNING | English target raised to 78–84 words (2.86 words per second on screen) |
| E7 | Uzbek script rejected: 34 words, asked 55–60 | the validator | per-scene word targets: **LLMs hit small numbers better** |
| E8 | Old `script.json` files would fail the new word rule | thinking ahead about `--run` | `Script.load()` with a validation context |
| E9 | One script call took 101 s (others ~3.5 s) | timing every step | **not explained**: maybe a free-tier queue or the SDK's silent retries. Left open |
| E10 | A copied run used the **original** run's audio, and crashed across drives | the traceback (`path is on mount 'D:', start on mount 'C:'`), then opening `voiceovers.json` | save paths **relative to the run folder**; a test copies a run and checks it uses its own audio |
| E11 | The app quit mid-check | the run was incomplete | re-ran it: same result, $0 |

**Open items:** measure a Russian run; E9; voicing is the slowest step (15–47 s): running the 6 voice
requests at once with `asyncio.gather` would make it a few seconds; no real FLUX image has gone through
yet; the 2× upscale jitter was judged from the maths and first/last frames, not on a phone yet.

---

## Check questions (optional, answer whenever you like)
**Images and voice**
1. Why does `mock_media()` compare with `!= "0"` and not `== "1"`?
2. In `_flux`, why is the ledger written *before* the download?
3. What's the difference between polling, subscribe and a webhook? Which one does `fal_client.subscribe` really do?
4. What does `await` actually do while the network is slow? What does `asyncio.run` do?
5. Why wait 1 s, then 2 s, before retrying? Why retry only `NoAudioReceived`?
6. What went wrong in the first version of the subtitle grouping, and how did a test catch it?

**ffmpeg and assembly**
7. In zoompan, what do `on`, `zoom`, `iw` and `x` mean? Why does `(iw-iw/zoom)/2` centre the window?
8. Why can't you pan at zoom 1.0? Why upscale the image 2× before zoompan?
9. Why are the scene clips silent, and why is each voice cut to exactly `frames × 1600` samples?
10. Why does a scene end at its last spoken word + 0.3 s, not at the end of its mp3?
11. What does `sidechaincompress` do to the music, and what are `[voice]` and `[key]` for?
12. What is −14 LUFS, and why normalise to it?
13. Why does ffmpeg run with `cwd=run_dir` and the plain name `subtitles.srt`?
14. Why ask the LLM for words per scene instead of seconds, and why did "per scene" fix the Uzbek script?
15. What was the bug with copied runs (E10), and how do relative paths fix it?

## Interview angles
- "Media generation sits behind one function with a free mock mode that's on unless explicitly
  disabled, and every paid job is priced before it runs and recorded the moment it's billed."
- "The audio decides the timing: LLMs can't estimate speech length, so scenes last as long as the real
  voice, and the LLM gets per-language word targets it can actually hit."
- "Picture, voice and subtitles share one list of integer frame counts (1600 audio samples per frame
  at 48 kHz / 30 fps), so they can't drift. I verified it by detecting speech onsets in the final mix."
- "Every ffmpeg command is built by a pure function and tested without running ffmpeg; only two live
  renders prove ffmpeg accepts them."
- "The final mix ducks the music under the voice with a sidechain compressor and normalises to −14 LUFS,
  the level the platforms play at."
- "A free, unofficial TTS dependency is wrapped behind one function with retry and exponential backoff,
  so a transient failure doesn't kill a run and the service can be swapped out later."
