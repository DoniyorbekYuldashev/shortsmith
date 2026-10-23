"""The Critic: frames from the rendered video + the script -> a CriticReport (Step 6.3).

    final.mp4 --ffmpeg--> 4 frames (one per chosen scene) --base64--> one vision call -> weak scenes + reasons

The Critic is the *evaluator* of the evaluator-optimizer loop (6.4): it only judges. Fixing is
the Reviser's job (reviser.py), and a different model writes than judges (the writer is
gemini-3.5-flash-lite, the critic gemini-3.8-flash), so a model never grades its own prose.

Code measures what code can (words per scene against the target, a hook copied from the notes or
the hook library) and hands the critic those facts: an LLM counts words badly, and a copied hook
is a string comparison, not an opinion. The critic judges what code can't: would the hook stop
a scroll, does the narration talk to the viewer or describe the picture, does each frame show
what its scene is about.

With MOCK_MEDIA=1 the frames are placeholder cards (the image prompt printed on a coloured card,
plus the burned-in subtitle). The critic is told so: it can read the prompt off the card and check
it against the narration, but it can't judge real image quality until real images exist.
"""

import base64
import os
import subprocess
from pathlib import Path

from shortsmith import llm
from shortsmith.agents import writer
from shortsmith.llm import ChatResult, Message
from shortsmith.media import ffmpeg
from shortsmith.media.tts import Voiceover
from shortsmith.models import NARRATION_WORDS, CriticReport, CriticVerdict, ResearchNotes, Script, ScriptChecks

# Stronger than the writer, and not the writer (Phase 5, D14). The free tier gives each model only 20
# requests a day, so SHORTSMITH_CRITIC_MODEL switches to a sibling (gemini-3.6-flash) when they run out.
CRITIC_MODEL = os.environ.get("SHORTSMITH_CRITIC_MODEL", "gemini-3.8-flash")
FRAMES = 4  # frames per review: a frame of every scene would cost ~50% more image tokens
FRAME_WIDTH = 540  # half size: enough to read a card or judge a composition, a quarter of the pixels

CRITIC_SYSTEM_PROMPT = """You are the critic of a ~30-second vertical short video (YouTube Shorts, Reels, \
TikTok). You get the script scene by scene, facts that code measured, and (when there are frames) still \
frames from the rendered video, each labelled with its scene. Find what would make a viewer scroll away \
and say which scenes to fix.

Score the video from 0 to 10 as a viewer would experience it (5 = average, 8 = ready to post, 10 = exceptional). Judge:
- hook: do the first 2 seconds stop the scroll? A hook copied word for word from somewhere is a weakness.
- narration: natural spoken language that talks TO the viewer (facts, tips, reasons, feelings). Narration \
that only describes what the picture shows ("warm lights glow softly") is a weakness: the picture shows it already.
- fit: does each scene's picture show what its narration is about?
- length: the measured word count against the target (too short = a video under 25 s, too long = over 35 s).
- flow: hook -> payoff -> call to action.

weak_scenes: at most 3, worst first, only scenes that clearly need a change. For each: problem = "narration" \
(change the words), "visual" (change the picture) or "both"; one sentence on what is wrong; one concrete fix. \
A video that scores 8 or more usually has no weak scenes.
cannot_judge: say plainly what these inputs don't let you judge."""

MOCK_NOTE = """The pictures are PLACEHOLDER CARDS (mock mode: no image model yet): a plain coloured card with \
the image prompt printed on it, plus the burned-in subtitle. Judge whether the prompt on the card fits the \
narration and the topic. Don't judge image quality, colours or composition, and never flag a scene only \
because it is a placeholder."""

SCRIPT_NOTE = "There are no frames: judge the script only (the picture fit from the image prompts)."


# --- facts code measures -----------------------------------------------------------------------


def checks(script: Script, notes: ResearchNotes | None = None) -> ScriptChecks:
    counts = [len(scene.narration.split()) for scene in script.scenes]
    low, high = NARRATION_WORDS[script.language]
    return ScriptChecks(words=counts, total_words=sum(counts), target=(low, high), in_range=low <= sum(counts) <= high,
                        hook_words=len(script.hook.split()),
                        hook_copied_from=writer.copied_from(script.hook, writer.hook_sources(notes)))


def facts_text(c: ScriptChecks) -> str:
    low, high = c.target
    verdict = "in range" if c.in_range else ("too short" if c.total_words < low else "too long")
    return (f"Measured by code: words per scene {c.words}; total {c.total_words} (target {low}-{high}: {verdict}); "
            f"hook {c.hook_words} words; hook copied word for word: {c.hook_copied_from or 'no'}")


# --- frames ----------------------------------------------------------------------------------------


def pick_scenes(indexes: list[int], focus: list[int] = (), k: int = FRAMES) -> list[int]:
    """Which scenes to look at: the ones just regenerated first (`focus`), then spread evenly over
    the video, which always includes the hook (first) and the call to action (last). In video order."""
    chosen = [i for i in focus if i in indexes][:k]
    n = len(indexes)
    spread = [indexes[round(j * (n - 1) / (k - 1))] for j in range(k)] if k > 1 else indexes[:1]
    for i in spread + list(indexes):
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen)


def frame_times(voiceovers: list[Voiceover], indexes: list[int]) -> dict[int, float]:
    """The middle of every scene on the final video's timeline (the editor's own frame counts)."""
    counts = ffmpeg.scene_frames(voiceovers)
    starts = ffmpeg.scene_starts(counts)
    return {idx: round(start + n / ffmpeg.FPS / 2, 3) for idx, start, n in zip(indexes, starts, counts, strict=True)}


def frame_cmd(video: Path, at_s: float, out: Path, width: int = FRAME_WIDTH) -> list[str]:
    """One JPEG frame at `at_s`. -ss before -i seeks fast; the frame is exact anyway (ffmpeg decodes from
    the keyframe before it)."""
    return ffmpeg.FFMPEG + ["-ss", f"{at_s:.3f}", "-i", video.as_posix(), "-frames:v", "1",
                            "-vf", f"scale={width}:-2", "-q:v", "3", out.as_posix()]


def sample_frames(run_dir: Path, voiceovers: list[Voiceover], script: Script, focus: list[int] = (),
                  round_: int = 0) -> list[tuple[int, float, Path]]:
    """(scene, time, jpeg) for the chosen scenes, saved in critic/round_N/ (kept: the build notes show them)."""
    indexes = [scene.idx for scene in script.scenes]
    times = frame_times(voiceovers, indexes)
    folder = run_dir / "critic" / f"round_{round_}"
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx in pick_scenes(indexes, focus):
        out = folder / f"scene_{idx:02d}.jpg"
        ffmpeg.run(frame_cmd(Path("final.mp4"), times[idx], out.relative_to(run_dir)), cwd=run_dir)
        frames.append((idx, times[idx], out))
    return frames


def data_url(jpeg: Path) -> str:
    """An image inside the request itself (no upload, no URL the API must fetch): base64 text."""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.read_bytes()).decode("ascii")


# --- the request -------------------------------------------------------------------------------------


def script_text(script: Script) -> str:
    # The hook is the start of scene 1, spoken once. Shown as a separate "Hook:" line, the critic read it
    # as said twice and flagged "scene 1 repeats the hook" (2 of 10 reviews in the 6.8 evaluation, E12).
    lines = [f"Title: {script.title}", f"Hook (the opening words of scene 1, spoken once): {script.hook}"]
    for scene in script.scenes:
        lines.append(f"Scene {scene.idx}: {scene.narration}")
        lines.append(f"  image prompt: {scene.visual_prompt}")
    return "\n".join(lines + [f"Call to action: {script.cta}"])


def critic_messages(topic: str, script: Script, facts: ScriptChecks, frames: list[tuple[int, float, Path]] = (),
                    mock: bool = True) -> list[Message]:
    """System = the rubric. User = one message with text parts and image parts, in the OpenAI format
    Gemini's compatible endpoint also reads ({"type": "image_url", "image_url": {"url": "data:..."}})."""
    note = (MOCK_NOTE if mock else "The pictures are real generated images.") if frames else SCRIPT_NOTE
    text = (f"Topic: {topic}\nLanguage: {llm.LANGUAGES[script.language]}\n\n{script_text(script)}\n\n"
            f"{facts_text(facts)}\n\n{note}")
    content: list[dict] = [{"type": "text", "text": text}]
    for idx, at_s, jpeg in frames:
        content.append({"type": "text", "text": f"Frame from scene {idx} ({at_s:.1f} s into the video):"})
        content.append({"type": "image_url", "image_url": {"url": data_url(jpeg)}})
    return [{"role": "system", "content": CRITIC_SYSTEM_PROMPT}, {"role": "user", "content": content}]


def review(topic: str, script: Script, notes: ResearchNotes | None = None, frames: list[tuple[int, float, Path]] = (),
           round_: int = 0, mock: bool = True, model: str | None = None) -> tuple[CriticReport, ChatResult]:
    """One critic round. Without frames it reviews the script alone (the 6.8 evaluation)."""
    model = model or CRITIC_MODEL
    facts = checks(script, notes)
    verdict, result = llm.paced(lambda: llm.chat_json(critic_messages(topic, script, facts, frames, mock),
                                                      CriticVerdict, model=model, step="critic"))
    indexes = {scene.idx for scene in script.scenes}
    issues = [issue for issue in verdict.weak_scenes if issue.scene in indexes]  # a scene that doesn't exist is dropped
    report = CriticReport(**verdict.model_dump(exclude={"weak_scenes"}), weak_scenes=issues, round=round_,
                          mode="video" if frames else "script", frames=[idx for idx, _, _ in frames],
                          model=result.model, checks=facts)
    return report, result


def run_frames(run_dir: Path, script: Script, voiceovers: list[Voiceover], focus: list[int], round_: int,
               ) -> list[tuple[int, float, Path]]:
    """sample_frames(), but a missing ffmpeg or video is an error with a clear message."""
    if not (run_dir / "final.mp4").exists():
        raise FileNotFoundError(f"{run_dir / 'final.mp4'} is missing: the editor runs before the critic")
    try:
        return sample_frames(run_dir, voiceovers, script, focus, round_)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"could not cut frames from final.mp4: {' '.join(getattr(e, '__notes__', []))}") from e
