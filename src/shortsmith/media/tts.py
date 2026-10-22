"""Voiceover with edge-tts: one audio file per scene, plus word timings for subtitles.

edge-tts uses the voice service behind Microsoft Edge's "Read aloud". It's free and needs
no key, but it's unofficial, so it stays behind `synthesize()` in case it ever has to go.
The audio decides the timing: each scene lasts as long as its spoken words, plus a short
pause the editor adds (ffmpeg.scene_frames).
"""

import argparse
import asyncio
import subprocess
import time
from pathlib import Path

import edge_tts
from pydantic import BaseModel, TypeAdapter

from shortsmith.models import Script

VOICES = {"en": "en-US-AndrewNeural", "uz": "uz-UZ-MadinaNeural", "ru": "ru-RU-SvetlanaNeural"}
TICKS_PER_SECOND = 10_000_000  # edge-tts reports times in 100-nanosecond "ticks"
ATTEMPTS = 3  # the free service sometimes answers with no audio; a moment later it usually works


class Word(BaseModel):
    text: str
    start_s: float
    end_s: float


class Voiceover(BaseModel):
    """One scene's audio: the file, its real length, and when each word is spoken."""

    audio_path: Path
    duration_s: float
    words: list[Word]


async def _speak(text: str, voice: str, out_path: Path) -> list[Word]:
    """Stream the audio into a file and collect word timings as they arrive."""
    communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")  # default is sentence timings
    words = []
    with out_path.open("wb") as audio:
        async for chunk in communicate.stream():  # each await = wait for the next network message
            if chunk["type"] == "audio":
                audio.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / TICKS_PER_SECOND
                end = start + chunk["duration"] / TICKS_PER_SECOND
                words.append(Word(text=chunk["text"], start_s=start, end_s=end))
    return words


def audio_duration(path: Path) -> float:
    """Exact length of an audio file in seconds, measured by ffprobe (installed with ffmpeg)."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,  # a non-zero exit code raises instead of returning garbage
    )
    return float(result.stdout)


def synthesize(text: str, voice: str, out_path: Path) -> Voiceover:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, ATTEMPTS + 1):
        try:
            words = asyncio.run(_speak(text, voice, out_path))  # runs the async function to completion
            break
        except edge_tts.exceptions.NoAudioReceived:
            if attempt == ATTEMPTS:
                raise  # give up loudly: a silent scene would break the video
            time.sleep(2 ** (attempt - 1))  # back off 1 s, then 2 s, instead of hammering the service
    return Voiceover(audio_path=out_path, duration_s=audio_duration(out_path), words=words)


def srt_time(seconds: float) -> str:
    """SRT time format, 00:01:02,345: note the comma before the milliseconds."""
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def group_words(words: list[Word], per_line: int = 4) -> list[list[Word]]:
    """Short subtitle lines (TikTok style). Call it per scene, so no line spans a scene cut."""
    return [words[i : i + per_line] for i in range(0, len(words), per_line)]


def to_srt(lines: list[list[Word]]) -> str:
    """Each line is shown from its first word's start to its last word's end."""
    blocks = []
    for number, line in enumerate(lines, start=1):
        text = " ".join(word.text for word in line)
        blocks.append(f"{number}\n{srt_time(line[0].start_s)} --> {srt_time(line[-1].end_s)}\n{text}\n")
    return "\n".join(blocks)


def shift(words: list[Word], by_s: float) -> list[Word]:
    """Move a scene's word times to where the scene starts in the whole video."""
    return [Word(text=w.text, start_s=w.start_s + by_s, end_s=w.end_s + by_s) for w in words]


def subtitles(voiceovers: list[Voiceover], scene_starts: list[float]) -> str:
    """One SRT for the whole video. The editor decides where scenes start (voice + pause),
    so the subtitles follow the final video's timeline, not the voice files laid end to end."""
    lines = []
    for voiceover, start in zip(voiceovers, scene_starts, strict=True):
        lines += group_words(shift(voiceover.words, start))  # grouped per scene: no line crosses a cut
    return to_srt(lines)


_VOICEOVERS = TypeAdapter(list[Voiceover])  # validates/serializes a plain list, like a model would
VOICEOVERS_FILE = "voiceovers.json"


def voice_script(script: Script, run_dir: Path) -> list[Voiceover]:
    """One audio file per scene, plus voiceovers.json with every word's timing.

    The JSON lets a run be re-assembled later without asking edge-tts again. Its audio paths
    are relative to the run folder, so a copied or moved run still finds its own files.
    """
    voice = VOICES[script.language]
    voiceovers = [
        synthesize(scene.narration, voice, run_dir / "audio" / f"scene_{scene.idx:02d}.mp3")
        for scene in script.scenes
    ]
    save_voiceovers(voiceovers, run_dir)
    return voiceovers


def save_voiceovers(voiceovers: list[Voiceover], run_dir: Path) -> None:
    """Write voiceovers.json, one entry per scene in order, with paths relative to the run folder."""
    saved = [v.model_copy(update={"audio_path": v.audio_path.relative_to(run_dir)}) for v in voiceovers]
    (run_dir / VOICEOVERS_FILE).write_bytes(_VOICEOVERS.dump_json(saved, indent=2))


def load_voiceovers(run_dir: Path) -> list[Voiceover] | None:
    """The saved voiceovers, or None if any part is missing (then the run is voiced again)."""
    path = run_dir / VOICEOVERS_FILE
    if not path.exists():
        return None
    saved = _VOICEOVERS.validate_json(path.read_bytes())
    voiceovers = [v.model_copy(update={"audio_path": run_dir / v.audio_path}) for v in saved]
    return voiceovers if all(v.audio_path.exists() for v in voiceovers) else None


if __name__ == "__main__":
    from shortsmith.media.ffmpeg import scene_frames, scene_starts  # demo only: the editor's timeline

    parser = argparse.ArgumentParser(description="Voice every scene of a run's script.json.")
    parser.add_argument("run_dir", nargs="?", help="a runs/<timestamp> folder (default: the newest one)")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else max(Path("runs").glob("*/script.json")).parent

    script = Script.load(run_dir / "script.json")
    voiceovers = voice_script(script, run_dir)
    (run_dir / "subtitles.srt").write_text(subtitles(voiceovers, scene_starts(scene_frames(voiceovers))), encoding="utf-8")

    print(f"{run_dir}  ({script.language}, voice {VOICES[script.language]})\n")
    print(f"{'scene':<7}{'planned':>9}{'real':>8}{'words':>7}")
    for scene, voiceover in zip(script.scenes, voiceovers):
        print(f"{scene.idx:<7}{scene.duration_s:>8.1f}s{voiceover.duration_s:>7.1f}s{len(voiceover.words):>7}")
    planned = sum(scene.duration_s for scene in script.scenes)
    real = sum(voiceover.duration_s for voiceover in voiceovers)
    print(f"{'total':<7}{planned:>8.1f}s{real:>7.1f}s\n\nwrote {run_dir / 'audio'} and {run_dir / 'subtitles.srt'}")
