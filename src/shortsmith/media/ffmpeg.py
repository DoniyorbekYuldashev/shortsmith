"""Video assembly with ffmpeg: Ken Burns clips, voice track, music bed, subtitles -> final.mp4.

Every command is built by a pure function that returns a list of arguments, so tests can
check it without running ffmpeg, and no shell ever parses it (paths with spaces stay safe).
`run()` is the only place that executes one. Commands run with cwd=run_dir and short
relative file names: the subtitles filter can't take "D:\\All codes\\..." without escaping.
"""

import json
import os
import re
import subprocess
from itertools import accumulate
from pathlib import Path

from pydantic import BaseModel

from shortsmith.media.tts import Voiceover, subtitles
from shortsmith.models import Motion, Script

WIDTH, HEIGHT, FPS = 1080, 1920, 30  # vertical full HD: what Shorts, Reels and TikTok expect
UPSCALE = 2  # zoompan moves in whole pixels; on a 2x image one step is half an output pixel, so less jitter
ZOOM = 1.2  # camera travel: at 1.2 the window shows 1/1.2 = 83% of the picture
SAMPLE_RATE = 48_000  # 48000 / 30 = 1600 audio samples per video frame, exactly
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS

PAUSE_S = 0.3  # a breath after each scene's last word
END_HOLD_S = 0.7  # the last scene holds a little longer, so the video doesn't stop on the last word

MUSIC_DB = -20  # music sits ~15 LU under the voice (edge-tts speaks at about -19 LUFS)
MUSIC_FADE_IN_S, MUSIC_FADE_OUT_S = 1.5, 2.5
LOUDNESS_LUFS = -14  # the loudness YouTube and TikTok play at; edge-tts alone is ~-19.5 LUFS

# Am - F - C - G, one chord every 4 s: the placeholder bed until a real track is supplied.
CHORDS = [(110.00, 220.00, 261.63, 329.63), (87.31, 174.61, 220.00, 261.63),
          (65.41, 130.81, 164.81, 196.00), (98.00, 196.00, 246.94, 293.66)]
CHORD_S = 4.0

# libass style for burned-in subtitles. PlayResX/Y make sizes and margins real video pixels
# (an SRT otherwise gets a 384x288 canvas). Colours are &HAABBGGRR, alpha 00 = opaque.
SUBTITLE_STYLE = {
    "PlayResX": WIDTH, "PlayResY": HEIGHT,
    "FontName": "Arial", "FontSize": 84, "Bold": 1,
    "PrimaryColour": "&H00FFFFFF", "OutlineColour": "&H00000000",
    "BorderStyle": 1, "Outline": 6, "Shadow": 0,
    "Alignment": 2,  # bottom centre (numpad layout)
    "MarginV": 560, "MarginL": 90, "MarginR": 90,  # bottom edge of the text ~70% down the frame
}

FFMPEG = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]  # quiet, overwrite outputs


def frames(seconds: float) -> int:
    """Whole frames for `seconds`: a video can only cut on a frame boundary."""
    return max(1, round(seconds * FPS))


def zoompan(motion: Motion, n_frames: int) -> str:
    """One zoompan filter. z, x and y are expressions ffmpeg evaluates for every output frame.

    `on` is the output frame number (0 .. n_frames-1), so on/last runs from 0 to 1. The
    visible window is iw/zoom wide; x and y are its top-left corner in the input image.
    """
    progress = f"on/{max(n_frames - 1, 1)}"  # max(): a 1-frame clip must not divide by zero
    centre_x, centre_y = "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    if motion == "zoom_out":
        z, x, y = f"{ZOOM:g}-{ZOOM - 1:g}*{progress}", centre_x, centre_y
    elif motion == "pan_right":  # the camera moves right: the window slides from the left edge to the right edge
        z, x, y = f"{ZOOM:g}", f"(iw-iw/zoom)*{progress}", centre_y
    elif motion == "pan_left":
        z, x, y = f"{ZOOM:g}", f"(iw-iw/zoom)*(1-{progress})", centre_y
    else:  # zoom_in, and ai_clip until real clips exist
        z, x, y = f"1+{ZOOM - 1:g}*{progress}", centre_x, centre_y
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={n_frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"


def ken_burns_cmd(image: Path, out: Path, duration_s: float, motion: Motion) -> list[str]:
    """ffmpeg arguments that turn one still image into a silent moving clip of `duration_s`."""
    n = frames(duration_s)
    big_w, big_h = WIDTH * UPSCALE, HEIGHT * UPSCALE
    video_filter = ",".join([
        # 576x1024 FLUX and 1080x1920 placeholders both become 2160x3840; another aspect ratio
        # is covered and cropped to 9:16 instead of squashed
        f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase:flags=lanczos",
        f"crop={big_w}:{big_h}",
        zoompan(motion, n),  # one input frame in, n frames out
    ])
    return FFMPEG + [
        "-i", image.as_posix(),
        "-vf", video_filter,
        "-frames:v", str(n),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",  # near-lossless: it's re-encoded once more at the end
        "-pix_fmt", "yuv420p",
        out.as_posix(),
    ]


def scene_frames(voiceovers: list[Voiceover]) -> list[int]:
    """How many frames each scene stays on screen: until its last spoken word, plus a pause.

    edge-tts files end in 0.4-0.9 s of silence (English ~0.37 s, Uzbek ~0.93 s), so the last word's
    end time is used, not the file length. Whole frames, so the picture, the voice and the
    subtitles all cut at the same instant.
    """
    counts = []
    for i, voiceover in enumerate(voiceovers):
        spoken = voiceover.words[-1].end_s if voiceover.words else voiceover.duration_s
        hold = PAUSE_S + (END_HOLD_S if i == len(voiceovers) - 1 else 0.0)
        counts.append(frames(spoken + hold))
    return counts


def scene_starts(counts: list[int]) -> list[float]:
    """Where each scene starts in the final video, in seconds."""
    return [start / FPS for start in accumulate([0] + counts[:-1])]


def music_bed_cmd(out: Path, duration_s: float) -> list[str]:
    """A soft placeholder chord pad, generated by ffmpeg itself: nothing is downloaded.

    Each chord fades in and out over 8 s and overlaps its neighbours by half (sin^2 windows
    add up to 1), so the pad never clicks or drops out at a chord change.
    """
    cycle = CHORD_S * len(CHORDS)
    voices = []
    for k, chord in enumerate(CHORDS):
        phase = f"mod(t+{cycle + CHORD_S / 2 - k * CHORD_S:g},{cycle:g})"  # 0 when chord k starts fading in
        window = f"lt({phase},{2 * CHORD_S:g})*pow(sin(PI*{phase}/{2 * CHORD_S:g}),2)"
        notes = "+".join(f"sin(2*PI*{f:g}*t)" for f in chord)
        voices.append(f"{window}*({notes})")
    expression = f"0.45*({'+'.join(voices)})"  # ~-14 LUFS, about as loud as a mastered song
    return FFMPEG + [
        "-f", "lavfi", "-i", f"aevalsrc='{expression}':s={SAMPLE_RATE}:d={duration_s:.3f}",
        "-af", "lowpass=f=1500,aecho=0.8:0.5:90|180:0.3|0.2",  # darker and a little roomier
        out.as_posix(),
    ]


def subtitle_style() -> str:
    return ",".join(f"{key}={value}" for key, value in SUBTITLE_STYLE.items())


def concat_list(clips: list[Path]) -> str:
    """The concat demuxer's list file: paths are relative to the list file itself."""
    return "".join(f"file '{clip.as_posix()}'\n" for clip in clips)


def final_cmd(
    clip_list: Path, voices: list[Path], counts: list[int], music: Path, srt: Path, out: Path
) -> list[str]:
    """One command: concatenated clips + voice track + ducked music + burned-in subtitles.

    Inputs: 0 = the clips (concat demuxer), 1..n = one voice file per scene, n+1 = the music.
    Every scene's voice is padded with silence, then cut to exactly its frames x 1600 samples,
    so the audio can never drift away from the picture.
    """
    if not re.fullmatch(r"[\w.-]+", srt.as_posix()):
        raise ValueError(f"subtitles path {srt} must be a plain name: run ffmpeg with cwd=run_dir")
    total = sum(counts) * SAMPLES_PER_FRAME
    total_s = sum(counts) / FPS
    music_in = len(voices) + 1
    stereo = f"aformat=sample_rates={SAMPLE_RATE}:channel_layouts=stereo"

    graph = [f"[0:v]subtitles={srt.as_posix()}:force_style='{subtitle_style()}'[v]"]
    for i, n in enumerate(counts, start=1):
        samples = n * SAMPLES_PER_FRAME
        graph.append(f"[{i}:a]{stereo},apad=whole_len={samples},atrim=end_sample={samples}[s{i}]")
    pieces = "".join(f"[s{i}]" for i in range(1, len(counts) + 1))
    graph.append(f"{pieces}concat=n={len(counts)}:v=0:a=1,asplit=2[voice][key]")  # key = a copy that steers the ducking
    graph.append(
        f"[{music_in}:a]{stereo},atrim=end_sample={total},volume={MUSIC_DB}dB,"
        f"afade=t=in:d={MUSIC_FADE_IN_S:g},afade=t=out:st={total_s - MUSIC_FADE_OUT_S:.3f}:d={MUSIC_FADE_OUT_S:g}[bed]"
    )
    # ducking: whenever the voice is louder than the threshold, the music is turned down further
    graph.append("[bed][key]sidechaincompress=threshold=0.03:ratio=4:attack=20:release=400[ducked]")
    graph.append(
        "[voice][ducked]amix=inputs=2:duration=first:normalize=0,"  # normalize=0: keep both levels as set
        f"loudnorm=I={LOUDNESS_LUFS}:TP=-1.5:LRA=11,aresample={SAMPLE_RATE}[a]"  # loudnorm works at 192 kHz inside
    )

    cmd = FFMPEG + ["-f", "concat", "-i", clip_list.as_posix()]
    for voice in voices:
        cmd += ["-i", voice.as_posix()]
    cmd += ["-stream_loop", "-1", "-i", music.as_posix()]  # a short track loops; atrim cuts it to length
    return cmd + [
        "-filter_complex", ";".join(graph),
        "-map", "[v]", "-map", "[a]",
        "-frames:v", str(sum(counts)),
        "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",  # yuv420p: what phones decode
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",  # index at the front, so playback starts before the whole file downloads
        out.as_posix(),
    ]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Execute one command. On failure the exception carries ffmpeg's own error message."""
    try:
        subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    except subprocess.CalledProcessError as e:
        e.add_note(f"ffmpeg said: {e.stderr.strip()[-2000:]}")
        raise


def _relative(path: Path, run_dir: Path) -> Path:
    return Path(os.path.relpath(path.resolve(), run_dir.resolve()))


def assemble(run_dir: Path, script: Script, voiceovers: list[Voiceover], music: Path | None = None) -> Path:
    """Build run_dir/final.mp4 from the scene images and voice files already in the run folder.

    Picture and sound are built apart and meet only in the last command, both cut to the
    same frame counts. `music` is an optional track of your own; without it a bed is generated.
    """
    counts = scene_frames(voiceovers)
    srt = Path("subtitles.srt")
    (run_dir / srt).write_text(subtitles(voiceovers, scene_starts(counts)), encoding="utf-8")

    (run_dir / "clips").mkdir(exist_ok=True)
    clips = []
    for scene, n in zip(script.scenes, counts, strict=True):
        clip = Path("clips") / f"scene_{scene.idx:02d}.mp4"
        image = Path("scenes") / f"scene_{scene.idx:02d}.png"
        run(ken_burns_cmd(image, clip, n / FPS, scene.motion), cwd=run_dir)
        clips.append(clip)
    clip_list = Path("clips.txt")
    (run_dir / clip_list).write_text(concat_list(clips), encoding="utf-8")

    if music is None:
        music = Path("music.wav")
        run(music_bed_cmd(music, sum(counts) / FPS), cwd=run_dir)
    else:
        music = music.resolve()  # a user file can live anywhere; input paths need no escaping

    voices = [_relative(voiceover.audio_path, run_dir) for voiceover in voiceovers]
    out = Path("final.mp4")
    run(final_cmd(clip_list, voices, counts, music, srt, out), cwd=run_dir)
    return run_dir / out


class VideoInfo(BaseModel):
    """What ffprobe says about a finished file: used for the summary and the live tests."""

    width: int
    height: int
    fps: float
    frames: int
    duration_s: float
    size_bytes: int
    video_codec: str
    pix_fmt: str
    audio_codec: str | None


def probe(path: Path) -> VideoInfo:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-of", "json", "-show_entries",
         "format=duration,size:stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,nb_frames",
         str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    num, den = video["r_frame_rate"].split("/")
    return VideoInfo(
        width=video["width"],
        height=video["height"],
        fps=int(num) / int(den),
        frames=int(video["nb_frames"]),
        duration_s=float(data["format"]["duration"]),
        size_bytes=int(data["format"]["size"]),
        video_codec=video["codec_name"],
        pix_fmt=video["pix_fmt"],
        audio_codec=audio["codec_name"] if audio else None,
    )
