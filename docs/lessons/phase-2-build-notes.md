# Phase 2 build notes (Steps 2.3–2.5, plus the commit of 2.1–2.2)

Written by the build session for the teaching session. Everything below happened on this
machine (Windows 11, Python 3.12, uv 0.13.1, ffmpeg 9.0.2 gyan.dev full build) on
2026-10-21/22. Money spent: **$0** (Gemini free tier, placeholder images, edge-tts, local ffmpeg).

---

## 1. What was built

### Files

| File | Status | What it is |
|---|---|---|
| `src/shortsmith/media/ffmpeg.py` | new | Ken Burns command builder, scene timeline, music bed, final mix, `run()`, `assemble()`, `probe()` |
| `src/shortsmith/make.py` | new | the end-to-end command: `python -m shortsmith.make "topic" --lang uz`, or `--run <folder>` to re-assemble |
| `src/shortsmith/media/tts.py` | changed (2.2 was uncommitted) | subtitle writing split out (`subtitles()`), `voiceovers.json` save/load |
| `src/shortsmith/models.py` | changed | total-seconds rule replaced by a per-language word-count rule; `Script.load()` for saved scripts |
| `src/shortsmith/llm.py` | changed | `WRITER_SYSTEM_PROMPT` now asks for 6 scenes and word counts per language, built from `models.NARRATION_WORDS` |
| `tests/test_ffmpeg.py` | new | 31 tests: command builders (pure), timeline, assemble with faked `run`, 2 live renders |
| `tests/test_make.py` | new | 7 tests: reuse of images/voice, summary + warning, prompt/validator agreement, CLI |
| `tests/test_tts.py`, `tests/test_models.py` | changed | new timeline/subtitles tests, word-count rule tests, copied-run test |
| `exercises/phase-2/ken_burns.py` + `test_ken_burns.py` | new | the 🧠 exercise (signature + hints, body `raise NotImplementedError`) |
| `pyproject.toml` | changed | `edge-tts` dependency (2.2) + `[tool.pytest.ini_options] testpaths = ["tests"]` |
| `.gitignore` | changed | `data/music/` (your own music for `--music`) |

### `media/ffmpeg.py`, function by function

**Constants.** `WIDTH, HEIGHT, FPS = 1080, 1920, 30`; `UPSCALE = 2`; `ZOOM = 1.2`;
`SAMPLE_RATE = 48_000` so `SAMPLES_PER_FRAME = 1600` exactly (48000 / 30); `PAUSE_S = 0.3`;
`END_HOLD_S = 0.7`; `MUSIC_DB = -20`; `LOUDNESS_LUFS = -14`; `SUBTITLE_STYLE` (a dict turned
into libass `force_style`); `FFMPEG = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]`.

**`frames(seconds) -> int`**: `max(1, round(seconds * 30))`. A video can only cut on a frame
boundary, so every duration in the pipeline is turned into whole frames once and reused.

**`zoompan(motion, n_frames) -> str`** and **`ken_burns_cmd(image, out, duration_s, motion) -> list[str]`**
(Step 2.3, the exercise). Real command for one Uzbek scene (182 frames, pan_left):

```
ffmpeg -hide_banner -loglevel error -y -i scenes/scene_02.png
  -vf "scale=2160:3840:force_original_aspect_ratio=increase:flags=lanczos,
       crop=2160:3840,
       zoompan=z='1.2':x='(iw-iw/zoom)*(1-on/181)':y='(ih-ih/zoom)/2':d=182:s=1080x1920:fps=30"
  -frames:v 182 -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p clips/scene_02.mp4
```

What each filter does:
- `scale=2160:3840:force_original_aspect_ratio=increase:flags=lanczos`: upscale to 2x the output.
  FLUX's 576x1024 and the placeholder's 1080x1920 both become 2160x3840. `increase` means "cover"
  (never letterbox); `lanczos` is a sharp resampling filter.
- `crop=2160:3840`: if an image ever isn't 9:16, the extra is cut off instead of squashing it.
- `zoompan`: takes **one** input frame and outputs `d` frames. For every output frame it evaluates
  three expressions: `z` (zoom), then `x`, `y` (top-left corner of the visible window, which is
  `iw/zoom` wide and `ih/zoom` high), crops that window and scales it to `s=1080x1920`.
  `on` = output frame number 0..d-1, so `on/(d-1)` is the progress 0→1. Motions:
  - `zoom_in`: `z = 1+0.2*on/(d-1)`, window centred: `x = (iw-iw/zoom)/2`, `y = (ih-ih/zoom)/2`
  - `zoom_out`: `z = 1.2-0.2*on/(d-1)`, centred
  - `pan_right`: `z = 1.2` fixed, `x = (iw-iw/zoom)*on/(d-1)` (left edge → right edge), y centred
  - `pan_left`: same with `(1-on/(d-1))`
  - `ai_clip`: falls back to zoom_in (no clips are generated yet)
  - `max(d-1, 1)` in Python avoids `on/0` for a 1-frame clip.
- Why upscale first: zoompan positions its crop window in **whole input pixels**. On a 2x image
  one pixel step is half an output pixel, so the slow drift jitters less. (Bigger, e.g. 4x, is
  smoother but slower; 2x was the brief's choice and looks fine.)
- `-frames:v n`: a guard so the clip has exactly n frames. `-crf 18 -preset veryfast`: near-lossless
  because these clips are re-encoded once more in the final pass. `-pix_fmt yuv420p`: a PNG would
  give 4:4:4 colour, which many phones and QuickTime can't decode.

**`scene_frames(voiceovers) -> list[int]`**: each scene lasts until its **last spoken word** +
0.3 s (the last scene + 1.0 s), in whole frames. See decision D1 for why not the file length.

**`scene_starts(counts) -> list[float]`**: running sum → where each scene starts (seconds).
Used for both the subtitles and (implicitly, via sample counts) the audio.

**`music_bed_cmd(out, duration_s)`**: a placeholder pad generated by ffmpeg, nothing downloaded:

```
ffmpeg ... -f lavfi -i "aevalsrc='0.45*( <chord 1> + <chord 2> + <chord 3> + <chord 4> )':s=48000:d=30.067"
           -af "lowpass=f=1500,aecho=0.8:0.5:90|180:0.3|0.2" music.wav
chord k  = lt(mod(t+18-4k,16),8)*pow(sin(PI*mod(t+18-4k,16)/8),2)*(sin(2*PI*f1*t)+...+sin(2*PI*f4*t))
```
- `-f lavfi -i "..."`: the "input" is a filter graph that *generates* audio instead of a file.
- `aevalsrc`: evaluates an expression per sample (`t` = time in seconds). Chords Am–F–C–G, 4 notes
  each (root an octave down for warmth), one chord every 4 s, cycle 16 s.
- Each chord is multiplied by a sin² window 8 s long that starts 2 s before its slot; neighbouring
  windows overlap by half and **sin² windows with 50% overlap add up to exactly 1**, so the pad
  never clicks or dips at a chord change. `mod(t+18-4k, 16)`: the +16 keeps the argument positive.
- `lowpass=f=1500`: removes harsh highs (softer). `aecho`: two short echoes (90 and 180 ms) = a bit of "room".
- `0.45` amplitude → bed measures **−14.7 LUFS**, peak −3.2 dBFS (about as loud as a mastered song),
  so a user's own track and the generated bed get the same treatment in the mix.

**`final_cmd(clip_list, voices, counts, music, srt, out)`**: one command does the concat, the
voice track, the music, the ducking, loudness and the subtitles. Real command (Uzbek run):

```
ffmpeg -hide_banner -loglevel error -y
  -f concat -i clips.txt                       # input 0: the 6 clips one after another
  -i audio/scene_01.mp3 ... -i audio/scene_06.mp3   # inputs 1..6: one voice file per scene
  -stream_loop -1 -i music.wav                 # input 7: music, looped if shorter than the video
  -filter_complex
[0:v]subtitles=subtitles.srt:force_style='PlayResX=1080,PlayResY=1920,FontName=Arial,FontSize=84,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=6,Shadow=0,Alignment=2,MarginV=560,MarginL=90,MarginR=90'[v];
[1:a]aformat=sample_rates=48000:channel_layouts=stereo,apad=whole_len=291200,atrim=end_sample=291200[s1];
  ... one chain per scene (291200 = 182 frames x 1600 samples) ...
[s1][s2][s3][s4][s5][s6]concat=n=6:v=0:a=1,asplit=2[voice][key];
[7:a]aformat=sample_rates=48000:channel_layouts=stereo,atrim=end_sample=1380800,volume=-20dB,afade=t=in:d=1.5,afade=t=out:st=26.267:d=2.5[bed];
[bed][key]sidechaincompress=threshold=0.03:ratio=4:attack=20:release=400[ducked];
[voice][ducked]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]
  -map [v] -map [a] -frames:v 863
  -c:v libx264 -preset medium -crf 21 -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart final.mp4
```

`clips.txt` (the concat demuxer's list; paths are relative to the list file):
```
file 'clips/scene_01.mp4'
...
file 'clips/scene_06.mp4'
```

Filter by filter:
- `-f concat -i clips.txt`: the **concat demuxer** reads the clips back to back as one stream.
  Works because all clips have identical codec settings. (Contrast: the `concat` **filter** below
  joins decoded audio streams.) Relative names with only letters/digits/`_`/`.`/`/` count as "safe",
  so `-safe 0` isn't needed.
- `[0:v]subtitles=...`: libass burns the SRT into the picture. `force_style` overrides the default
  style. **`PlayResX/PlayResY=1080/1920`** make FontSize and margins real pixels (an SRT otherwise
  gets a 384x288 canvas and everything scales from that). Colours are `&HAABBGGRR` (alpha first,
  then blue-green-red; `00` = opaque). `BorderStyle=1` = outline + shadow (not a box), `Outline=6` px
  black. `Alignment=2` = bottom centre (numpad layout), `MarginV=560` = the text's bottom edge sits
  560 px above the bottom, ~70% down the frame, clear of the platforms' buttons and captions.
- `aformat=sample_rates=48000:channel_layouts=stereo`: edge-tts gives 24 kHz mono MP3; everything
  is converted to one format before joining and mixing.
- `apad=whole_len=N` then `atrim=end_sample=N`: pad with silence to *at least* N samples, then cut
  to *exactly* N. N = scene frames × 1600, so each scene's audio is exactly as long as its picture.
- `concat=n=6:v=0:a=1`: join the 6 padded voice pieces (0 video, 1 audio stream).
- `asplit=2[voice][key]`: two copies of the voice: one to hear, one to steer the ducking.
- music chain: `atrim` to the video length (it was looped), `volume=-20dB`, fade in 1.5 s, fade out
  over the last 2.5 s.
- `sidechaincompress` (= ducking): compresses the first input (music) based on the loudness of the
  second (the voice copy). When the voice is above `threshold=0.03` (≈ −30 dBFS) the music is turned
  down with `ratio=4`; `attack=20` ms (fast down), `release=400` ms (slow back up in the pauses).
- `amix=inputs=2:duration=first:normalize=0`: add voice + ducked music. `normalize=0` stops amix
  from dividing each input by the number of inputs (which would make the voice 6 dB quieter).
- `loudnorm=I=-14:TP=-1.5:LRA=11`: EBU R128 loudness normalisation to −14 LUFS (what YouTube and
  TikTok play at), true peak ≤ −1.5 dBTP. It works at 192 kHz internally, so `aresample=48000` follows.
- Output: `-frames:v 863` (exact length), H.264 `yuv420p` (phones), AAC 160k, `+faststart` (moves the
  MP4 index to the front so playback starts before the whole file has downloaded).

**`run(cmd, cwd=None)`**: `subprocess.run(..., capture_output=True, text=True, encoding="utf-8",
errors="replace", check=True)`; on `CalledProcessError` it attaches ffmpeg's own stderr with
`e.add_note(...)` (Python 3.11+) and re-raises, so the traceback shows *why* ffmpeg failed.

**`assemble(run_dir, script, voiceovers, music=None) -> Path`**: the Editor. Computes the frame
counts, writes `subtitles.srt` on the video timeline, renders one clip per scene, writes
`clips.txt`, generates `music.wav` (unless `--music` was given), runs the final command. All
commands run with `cwd=run_dir` and short relative names (decision D3).

**`probe(path) -> VideoInfo`**: ffprobe JSON → a Pydantic model (width, height, fps, frames,
duration, size, codecs). Used by the summary and the live tests.

### `media/tts.py` changes
- `voice_script(script, run_dir)` now only voices; it no longer writes the SRT (the editor owns the
  timeline). It saves `voiceovers.json` (every word's timing) with audio paths **relative to the run
  folder** (see error E10).
- `subtitles(voiceovers, scene_starts) -> str`: per scene, shift words to the scene's start and
  group them into 4-word lines (no line crosses a cut).
- `load_voiceovers(run_dir)`: reads the JSON back (via `TypeAdapter(list[Voiceover])`), resolves
  paths against `run_dir`, returns `None` if anything is missing (→ voice again; edge-tts is free).

### `models.py` / `llm.py` changes (the Step 2.2 duration problem)
- Removed `TOTAL_SECONDS = (20, 40)` (it checked the LLM's `duration_s` guesses, which ran ~40% long).
- Added `NARRATION_WORDS = {"en": (78, 84), "uz": (55, 60), "ru": (58, 64)}` and `WORDS_SLACK = 0.25`:
  the validator rejects a script only if its total narration words are outside
  low×0.75 … high×1.25 (EN 58–105, UZ 41–75, RU 43–80). The post-voice summary warns outside 25–35 s.
- `SCENES_PLANNED = 6` (matches the 6-images-per-video target). The validator still accepts 4–8.
- `duration_s` stays in the schema as the LLM's *plan*, printed next to the real numbers.
- `Script.load(path)`: validates a saved `script.json` with `context={"saved": True}`; the
  `model_validator` (now taking `info: ValidationInfo`) skips the word-count rule for saved scripts
  but still checks structure. Reason: a script written under an older rule must stay re-assemblable.
- `WRITER_SYSTEM_PROMPT` is now an f-string built from `NARRATION_WORDS`, so prompt and validator
  can't disagree (a test checks it). New rules: exactly 6 scenes; per-language words **per scene**
  and in total; the cta is spoken at the end of the last scene; narration has no emojis/hashtags;
  `duration_s` is "your estimate".

### `make.py`
`main()`: parse args → (topic) `new_run()` = the only LLM call, saves `script.json` → `Script.load`
→ `make_images()` (skips existing images: never pay twice) → `make_voice()` (reuses
`voiceovers.json` if it matches the script) → `ffmpeg.assemble()` → `summary()` table
(planned / voice file / on screen / words), ffprobe info, a WARNING outside 25–35 s, run time, this
run's ledger cost (ledger total after − before). A small `step()` context manager prints each
stage with its duration. A `ValidationError` from the writer prints the broken rules and exits 1
(the call stays in the ledger).

---

## 2. Results (real numbers)

### Final demo videos (both $0)

| | English | Uzbek |
|---|---|---|
| Command | `make "coffee shop in Tashkent, cozy autumn vibe" --lang en` | `make "Samarqand, Registon maydoni kechqurun" --lang uz` |
| Output | `runs/2026-10-22_112855/final.mp4` | `runs/2026-10-22_112757/final.mp4` |
| Length | **30.1 s** (902 frames) | **28.8 s** (863 frames) |
| Words | 75 (target 78–84) | 56 (target 55–60) |
| File | 2.8 MB, 1080x1920, 30 fps, h264/yuv420p + aac 48 kHz stereo | 2.6 MB, same |
| Loudness | −14.3 LUFS, LRA 2.5 LU, true peak −1.5 dBFS | −14.7 LUFS, LRA 3.0 LU, peak −1.6 dBFS |
| Stage times | script 3.5 s, images 0.6 s, voice 46.6 s, ffmpeg 28.2 s; **79 s** total | script 3.6 s, images 0.7 s, voice 15.2 s, ffmpeg 27.4 s; **47 s** total |
| Ledger (list price) | $0.001631 (362 in / 609 out tokens) | $0.001881 (370 in / 708 out) |

Per scene, English (planned = LLM guess, voice = mp3 length, on screen = last word + pause):

| scene | motion | planned | voice | on screen | words |
|---|---|---|---|---|---|
| 1 | zoom_in | 5.0 | 5.1 | 5.1 | 12 |
| 2 | pan_left | 5.0 | 6.1 | 6.1 | 12 |
| 3 | zoom_out | 5.0 | 4.9 | 4.8 | 13 |
| 4 | pan_right | 5.0 | 4.5 | 4.4 | 13 |
| 5 | zoom_in | 5.0 | 4.8 | 4.8 | 13 |
| 6 | zoom_out | 5.0 | 4.2 | 4.9 | 12 |
| total | | 30.0 | 29.7 | 30.1 | 75 |

Per scene, Uzbek:

| scene | motion | planned | voice | on screen | words |
|---|---|---|---|---|---|
| 1 | zoom_in | 5.0 | 6.7 | 6.1 | 9 |
| 2 | pan_left | 5.0 | 5.5 | 4.9 | 9 |
| 3 | zoom_out | 5.0 | 4.8 | 4.2 | 10 |
| 4 | pan_right | 5.0 | 5.7 | 5.1 | 9 |
| 5 | zoom_in | 5.0 | 4.5 | 3.8 | 11 |
| 6 | zoom_out | 5.0 | 4.6 | 4.7 | 8 |
| total | | 30.0 | 31.8 | 28.8 | 56 |

With word targets, the plan and reality now agree within ~1 s (Step 2.2: planned ~29 s vs ~17–18 s real).
Note how "on screen" is *shorter* than "voice" for Uzbek: each Uzbek file ends in ~0.93 s of silence
that is cut (D1). Both scripts start with the hook and end with the spoken cta
(EN: "Craving the ultimate cozy autumn vibe?" … "Save this reel and visit this weekend.").

### Measured voice facts (edge-tts)
- Silent tail per file: English (AndrewNeural) 0.36–0.38 s; Uzbek (MadinaNeural) 0.93–0.95 s.
  Lead-in silence ~0.15 s (EN) / ~0.21 s (UZ).
- Speaking rate *per second on screen* (pauses included): EN 75 words / 30.1 s ≈ 2.5–2.9 w/s
  (first EN run: 65 / 22.7 s = 2.86); UZ 56 / 28.8 s ≈ 1.94 w/s, and 30 / 15.2 s = 1.97 on the
  Step 2.2 run. Russian was **not** measured (target 58–64 is an estimate).
- Loudness of an edge-tts file: −19.5 LUFS integrated, true peak −3.8 dBFS.

### Sync check (does the voice match the subtitles?)
`silencedetect=n=-22dB:d=0.1` on the final mix found speech onsets at scene starts; compared
with the subtitle start times from the same timeline:

| English scene | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| subtitle starts | 5.17 | 11.25 | 16.10 | 20.48 | 25.27 |
| speech detected | 5.22 | 11.30 | 16.15 | 20.45 | 25.20/25.32 |

Uzbek: 6.20→6.27, 11.12→11.18, 15.30→15.37. The offset is a constant ~+0.05 s (the detector
needs a few ms of loud audio) and **does not grow** from scene 2 to scene 6: no drift.

### Mix levels (prototype on the Step 2.2 Uzbek run)
Mean level: speech −17 to −18 dB, music alone in the gap between scenes −32 dB (ducked ~4 dB below
its undocked level), music in the fade-out tail −43 dB. Voice ≈ 15 dB above the music.

### Other runs
- Prototype re-assembly of the Step 2.2 Uzbek run (30 words, old prompt): 15.2 s, 455 frames,
  1.3 MB; first assembly 16.4 s, voice 25.0 s.
- First English run with the first word targets (70–75): `runs/2026-10-21_210620`, 65 words →
  22.7 s → WARNING; script call took **101.0 s** (see E9), voice 38.1 s, ffmpeg 28.8 s, 169 s total,
  $0.001610.
- Rejected Uzbek call (34 words): $0.001607 list price, in the ledger at 2026-10-21T16:08:31Z.
- Re-assembly without LLM: English demo re-voiced (old-format voiceovers.json) and rebuilt in 54 s
  with identical timings (edge-tts gives the same audio for the same text); a copy of the Uzbek run
  in another folder + `--music "D:\All codes\shortsmith\runs\...\music.wav"` (a path with a space):
  34 s, voice 0.0 s (reused), no `music.wav` generated, 0 new ledger lines.
- Ledger total after the phase: $0.013472 list price; really billed $0 (Gemini free tier).
- Scene clips for a 30 s video: ~2.4–2.7 MB; `music.wav` ~2.8–2.9 MB (gitignored).

### Tests
- `uv run pytest`: **76 passed** (~4–15 s; the two live 1-second renders take ~0.5 s each).
  Per file: test_ffmpeg 31, test_tts 10, test_models 9, test_ledger 8, test_llm 7, test_make 7, test_images 4.
- `uv run pytest exercises/phase-2`: 16 tests, all fail with `NotImplementedError: your turn: follow
  the hints above`. The same 16 tests pass when `ken_burns.py` just re-exports the real function
  (checked in a scratch folder), so the exercise is solvable as written.

---

## 3. Errors hit, how they were found, how they were fixed

**E1. Shell heredoc ate my backslashes (tooling).** Editing `tts.py` with a Python script inside a
Git Bash heredoc turned `\\n` inside f-strings into real line breaks, so two `print(f"...")` lines were
split across lines (would be a `SyntaxError: unterminated string`). Found by reading the file back
right after the edit. Fixed with the Edit tool; a second heredoc edit (tests) failed its own
`assert old in s` for the same reason, so after that all edits went through the Edit tool.
Lesson: escaping changes at every layer (shell → Python → ffmpeg filter → libass).

**E2. Music bed inaudible.** The first bed (`0.12*(...)`) measured `I: -25.6 LUFS`, peak −14 dBFS
(ffmpeg `ebur128`). At `volume=-20dB` that is ~−46 LUFS, about 26 LU under the voice: gone. Found by
measuring, not by listening. Fix: amplitude 0.45 → −14.7 LUFS standalone, i.e. ~−34 LUFS in the mix.

**E3. Whole video too quiet.** Mix measured `I: -20.9 LUFS`, peak −8.9 dBFS; platforms play at ~−14
and YouTube doesn't turn quiet videos *up*. Fix: `loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000` at
the end of the audio graph → −14.3 / −14.7 LUFS. (`aresample` because loudnorm outputs 192 kHz.)

**E4. ebur128 flooded the terminal (tooling).** `-af ebur128` prints a line every 100 ms at info
level (hundreds of lines). Fix: filter the output for the summary lines (`I:`, `LRA:`, `Peak:`).

**E5. Two of my own tests were wrong.** `uv run pytest` → 2 failed, e.g.
`AssertionError: assert ['audio/scene...', '-1', '-i'] == ['-stream_loop', '-1', '-i']`. The slice
`cmd[i-4:i]` takes 4 items but the expected list has 3. Fixed to `cmd[i-3:i]`. (Two more test
mistakes were caught by re-reading before the first run: an expected SRT end of `04,000` that should
be `04,300`, and audio paths on drive D: with `tmp_path` on drive C:, which `os.path.relpath` can't
relate.)

**E6. First English run 7 s short.** `WARNING: 22.7 s is outside the 25-35 s target`. The LLM wrote
65 words (asked 70–75) and the voice speaks 2.86 words per on-screen second once silent tails are
trimmed (D1), so 70–75 words can never reach 30 s. Fix: English target 78–84 (see E7 for the prompt).

**E7. Uzbek script rejected.** `The model's script broke 1 rule(s) ... Value error, narration has 34
words in total, expected about 55-60 (uz, ~30 s)`. Gemini 3.5 Flash-Lite undershoots a *total* word
count badly, worst in Uzbek. Fix in the prompt: "Exactly 6 scenes" and words **per scene** next to the
total ("Uzbek (Latin script): 9-10 words per scene, 55-60 words in total"). An LLM can hit 10 words
far better than 60. Next runs: Uzbek 56 words, English 75 words, both inside 25–35 s.

**E8. Word-count rule would block old runs.** Old `script.json` files (30 and 44 words) fail the new
rule, so `--run` on them would crash in validation. Fix: `Script.load()` passes a validation
context and the rule is skipped for saved scripts (structure still checked; test added).

**E9. One writer call took 101.0 s** (first English run); every later call took 3.5–3.6 s. Not
diagnosed: a run with `httpx`/`openai` INFO logging was fast and showed no retry lines. Likely a
free-tier queue or the OpenAI SDK's silent retries (default `max_retries=2`, waiting on
`retry-after`). Left open.

**E10. A copied run used the wrong audio / crashed (real bug).** Re-assembling a copy of the Uzbek
run in the scratch folder failed with
`ValueError: path is on mount 'D:', start on mount 'C:'` (in `ffmpeg._relative` → `os.path.relpath`).
Found from the traceback, then by opening the copy's `voiceovers.json`: it held
`"audio_path": "runs\\2026-10-22_112757\\audio\\scene_01.mp3"`, a path relative to the **repo root**.
So any copied or moved run silently points back at the *original* run's audio (and across drives it
crashes). Fix: `voice_script` saves paths relative to the run folder (`audio\\scene_01.mp3`),
`load_voiceovers` joins them with `run_dir`. Test `test_a_copied_run_folder_uses_its_own_audio`.
Old-format files no longer resolve, so those runs are voiced again on `--run` (free).

**E11. App quit mid-check.** The session ended during the rebuild of both demo runs; the Uzbek one had
finished, the English one had not. Re-ran it: same timings, $0.

---

## 4. Design decisions and why

**D1. A scene ends at its last spoken word + 0.3 s, not at the end of its mp3.** The brief said
"voice duration + ~0.3 s", but the measurements showed edge-tts files end in 0.36 s (EN) and 0.93 s
(UZ) of silence. File length + 0.3 s would give ~1.4 s gaps between Uzbek scenes (sluggish for a
short). The word-boundary end comes ~0.1 s before the real end of the sound, which the 0.3 s pause
covers, so no speech is cut. The last scene holds 0.7 s longer so the video and the music fade don't
stop on the last syllable.

**D2. Picture and sound are built separately and meet only in the final command, both cut to the
same whole frames.** Per-scene clips with their own AAC audio, joined by stream copy, would add
AAC priming/padding (~20–40 ms) at every join → cumulative drift and clicks. Instead: silent
clips; the voice is padded/cut per scene to frames × 1600 samples (integer arithmetic, no float
rounding) and joined with the `concat` filter; subtitles are shifted by the same frame counts.
Video, voice and subtitles can't drift because they share one list of integers.

**D3. ffmpeg runs with `cwd=run_dir` and relative names.** The `subtitles=` filter takes a file
name inside a filter graph, where `:` separates options and `\` and `'` are escape characters; a
Windows path like `D:\All codes\shortsmith\runs\...\subtitles.srt` would need several escaping
layers. With cwd = the run folder it's just `subtitles.srt`, and `final_cmd` refuses anything that
isn't a plain relative name. Inputs (`-i`) are ordinary arguments, so a user's music path can stay
absolute even with spaces. Commands are lists (never `shell=True`), so spaces are never split.

**D4. `-preset veryfast -crf 18` for scene clips, `-preset medium -crf 21` for the final.**
Intermediates are near-lossless and quick; only the final file is tuned for size (2.6–2.8 MB for
30 s).

**D5. Music: generated, never downloaded; optional `--music <file>`.** Brief rule. The bed is made
as loud as a mastered song (−14.7 LUFS) so one `volume=-20dB` works for both the bed and a real
track. A short track loops (`-stream_loop -1`), `atrim` cuts it. `data/music/` is gitignored for
licensed tracks.

**D6. Ducking with `sidechaincompress`.** Only two extra filters (`asplit` + `sidechaincompress`)
for a clearly better result: music dips under speech and comes back in pauses (release 400 ms).

**D7. `loudnorm` to −14 LUFS.** Single pass is enough for speech + a quiet bed (LRA 2.5–3.0 LU).
Two-pass loudnorm would be more exact but needs a measuring run first; not worth it now.

**D8. Subtitles: `PlayResX/Y` = video size, Arial 84 px bold, white with a 6 px black outline,
bottom-centre, `MarginV=560`.** Big and readable, above the platforms' UI. Uzbek apostrophes
(o', g') render fine in Arial. 4-word lines often wrap into two lines at this size, which is normal
for shorts.

**D9. Word targets, not seconds; per scene, not only in total; exactly 6 scenes.** Seconds come from
the audio now. Code counts words (`len(narration.split())`); the LLM gets small per-scene numbers it
can actually hit. Validator slack ±25% rejects only nonsense; the 25–35 s window is a warning
after voicing, not an error, because the real length is known only then.

**D10. The cta is spoken.** Before, `cta` was a separate field that was never voiced or shown. Now
the last scene's narration ends with it, like the hook opens scene 1.

**D11. `voiceovers.json` + `--run`.** Word timings are saved so a run can be re-edited (new music,
new subtitle style, a fixed scene) with no LLM and no edge-tts call. Images and voice are only made
if missing: in paid mode an image is never bought twice.

**D12. The Ken Burns tests evaluate the expressions instead of comparing strings.** `eval()` runs the
zoompan expressions for every frame with `on`, `zoom`, `iw`, `ih` bound, then checks where the camera
goes (starts at 1.0, ends at 1.2, stays centred, pans edge to edge, never leaves the picture). Any
correct formula passes, which the exercise needs. Python and ffmpeg agree on `+ - * / ( )`; the
exercise hints warn against ffmpeg-only syntax (`if()`, `^`).

**D13. Two live tests only.** Rendering a 1-second clip from a 576x1024 and a 1080x1920 image proves
ffmpeg accepts the command and that both FLUX and placeholder sizes give 1080x1920, 30 frames,
h264/yuv420p. Skipped if ffmpeg isn't on PATH. Everything else is pure or uses a faked `run()`.

---

## 5. Open items
- Russian word target (58–64) is unmeasured: do one Russian run and adjust `NARRATION_WORDS["ru"]`.
- E9: the one 101-s writer call is unexplained.
- Voicing is the slowest step (15–47 s for 6 scenes, one edge-tts request after another); running the
  6 requests concurrently (`asyncio.gather`) would cut it to a few seconds.
- No real FLUX image went through the pipeline yet ($0 rule); the 576x1024 path is covered by the
  live test with a generated image of that size.
- Jitter of the 2x upscale was judged from first/last frames and the maths, not by watching on a phone.
