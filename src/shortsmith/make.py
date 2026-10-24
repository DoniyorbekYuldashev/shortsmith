"""Phase 2 end to end, no agents yet: topic -> script -> images -> voice -> runs/<id>/final.mp4.

Phase 7's `shortsmith` command (cli.py) is the friendlier front end: `shortsmith make "topic"` runs
the graph, `shortsmith make "topic" --simple` this straight pipeline. This module stays as it was, so
the commands in the Phase 2-6 lessons keep working.

Run from the repo root:
    uv run --env-file .env python -m shortsmith.make "coffee shop in Tashkent, cozy autumn vibe" --lang uz
    uv run --env-file .env python -m shortsmith.make "coffee shop in Tashkent" --research --brand anor-va-dolchin
    uv run --env-file .env python -m shortsmith.make --run runs/2026-10-21_201504
    uv run --env-file .env python -m shortsmith.make "coffee shop in Tashkent" --graph --research --approve yes

--research (Phase 5) first looks at real YouTube trends and retrieves fitting hook patterns,
and the writer gets the Researcher's notes; --brand adds a brand's guideline. The last form
rebuilds final.mp4 in an existing run folder without any LLM call: it reuses the script,
images and voice files, and only makes the ones that are missing. --graph (Phase 6) runs the
LangGraph pipeline instead (agents/graph.py): researcher, writer (3 options + pick), director, an
approval step before the hero clip, producer, editor and a critic loop, checkpointed after every node.
"""

import argparse
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from shortsmith import ledger
from shortsmith.agents import researcher
from shortsmith.llm import LANGUAGES, write_script
from shortsmith.media import ffmpeg, images, tts
from shortsmith.models import Script

TARGET_SECONDS = (25.0, 35.0)  # a ~30-second short; outside this the word targets need tuning


@contextmanager
def step(name: str) -> Iterator[None]:
    """Print what's happening and how long it took, one line per pipeline step."""
    print(f"{name}...", end=" ", flush=True)
    start = time.perf_counter()
    yield
    print(f"{time.perf_counter() - start:.1f} s")


def new_run(topic: str, language: str, research: researcher.Research | None = None,
            run_dir: Path | None = None) -> Path:
    """Write a script (one LLM call) and save it in a fresh run folder, with the research behind it.
    `run_dir`: a folder name chosen first (`shortsmith make --simple` tags the ledger with it)."""
    script, _ = write_script(topic, language, research=research.notes if research else None)
    run_dir = run_dir or Path("runs") / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True)
    (run_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    if research:
        (run_dir / "research.json").write_text(research.model_dump_json(indent=2), encoding="utf-8")
    return run_dir


def make_images(script: Script, run_dir: Path) -> None:
    """One image per scene, skipping any that already exist: an image is never paid for twice."""
    for scene in script.scenes:
        path = run_dir / "scenes" / f"scene_{scene.idx:02d}.png"
        if not path.exists():
            images.generate_image(scene.visual_prompt, path, f"Scene {scene.idx} - {scene.motion}")


def make_voice(script: Script, run_dir: Path) -> list[tts.Voiceover]:
    """Reuse the saved voiceovers if they match the script; otherwise voice it (edge-tts is free)."""
    voiceovers = tts.load_voiceovers(run_dir)
    if voiceovers is None or len(voiceovers) != len(script.scenes):
        voiceovers = tts.voice_script(script, run_dir)
    return voiceovers


def summary(script: Script, voiceovers: list[tts.Voiceover], final: Path) -> str:
    """Planned (the LLM's guess) next to real (the voice file) and what's on screen."""
    counts = ffmpeg.scene_frames(voiceovers)
    rows = [f"{'scene':<7}{'motion':<11}{'planned':>8}{'voice':>8}{'on screen':>11}{'words':>7}"]
    for scene, voiceover, n in zip(script.scenes, voiceovers, counts, strict=True):
        rows.append(
            f"{scene.idx:<7}{scene.motion:<11}{scene.duration_s:>7.1f}s{voiceover.duration_s:>7.1f}s"
            f"{n / ffmpeg.FPS:>10.1f}s{len(scene.narration.split()):>7}"
        )
    planned = sum(scene.duration_s for scene in script.scenes)
    voice = sum(voiceover.duration_s for voiceover in voiceovers)
    words = sum(len(scene.narration.split()) for scene in script.scenes)
    rows.append(f"{'total':<18}{planned:>7.1f}s{voice:>7.1f}s{sum(counts) / ffmpeg.FPS:>10.1f}s{words:>7}")

    info = ffmpeg.probe(final)
    rows.append(
        f"\n{final}  {info.width}x{info.height}, {info.fps:g} fps, {info.duration_s:.1f} s, "
        f"{info.size_bytes / 1e6:.1f} MB, {info.video_codec}/{info.pix_fmt} + {info.audio_codec}"
    )
    low, high = TARGET_SECONDS
    if not low <= info.duration_s <= high:
        rows.append(f"WARNING: {info.duration_s:.1f} s is outside the {low:g}-{high:g} s target")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="One topic -> runs/<id>/final.mp4 (Phase 2: no agents yet).")
    parser.add_argument("topic", nargs="?", help="what the video is about")
    parser.add_argument("--lang", choices=sorted(LANGUAGES), default="en")
    parser.add_argument("--run", type=Path, help="rebuild final.mp4 in this existing run folder, no LLM call")
    parser.add_argument("--music", type=Path, help="your own music file (e.g. in data/music/); default: a generated bed")
    parser.add_argument("--research", action="store_true", help="ground the script in YouTube trends and retrieved hook patterns")
    parser.add_argument("--brand", help="write for a brand in data/hooks/brands/ (e.g. anor-va-dolchin); implies --research")
    graph = parser.add_argument_group("the LangGraph pipeline (Phase 6)")
    graph.add_argument("--graph", action="store_true", help="run the multi-agent graph: writer options, director, critic loop")
    graph.add_argument("--resume", metavar="RUN_ID", help="continue a graph run from its last checkpoint")
    graph.add_argument("--approve", choices=["yes", "no"], help="answer the hero-clip question without asking")
    graph.add_argument("--max-rounds", type=int, default=2, help="critic -> revise rounds at most (default 2)")
    graph.add_argument("--pass-score", type=float, default=7.0, help="critic score at which the video is done (default 7)")
    graph.add_argument("--prompt", choices=["v1", "v2"], default="v2", help="writer prompt version (default v2)")
    graph.add_argument("--mcp", action="store_true", help="the producer calls the media tools through the MCP server")
    graph.add_argument("--no-trace", action="store_true", help="don't record the run in MLflow")
    graph.add_argument("--crash-before", metavar="NODE", help="demo: kill the process when this node starts (resume proof)")
    args = parser.parse_args()
    if args.resume:
        if args.topic or args.run:
            parser.error("--resume continues a run: give no topic and no --run")
        from shortsmith.agents import graph as run_graph  # LangGraph is imported only when it is used
        return run_graph.run_cli(args)
    if args.graph:
        if args.topic is None or args.run:
            parser.error("--graph needs a topic (and no --run)")
        from shortsmith.agents import graph as run_graph
        return run_graph.run_cli(args)
    if (args.topic is None) == (args.run is None):
        parser.error("give either a topic or --run <run folder>")
    if args.music and not args.music.is_file():
        parser.error(f"music file not found: {args.music}")
    if args.run and (args.research or args.brand):
        parser.error("--research and --brand are for a new topic, not --run")

    start, cost_before = time.perf_counter(), ledger.total_cost(ledger.DEFAULT_PATH)
    if args.topic:
        research = None
        if args.research or args.brand:
            with step("researching (YouTube trends + hook library)"):
                research = researcher.research(args.topic, args.lang, args.brand)
            print(researcher.summary(research))
        try:
            with step(f"writing the script ({LANGUAGES[args.lang]})"):
                run_dir = new_run(args.topic, args.lang, research)
        except ValidationError as e:
            print(f"\nThe model's script broke {e.error_count()} rule(s) (the call is still in the ledger):")
            for error in e.errors():
                print(" -", error["msg"])
            raise SystemExit(1)
    else:
        run_dir = args.run
    script = Script.load(run_dir / "script.json")

    with step(f"images ({'mock placeholders' if images.mock_media() else 'FLUX, paid'})"):
        make_images(script, run_dir)
    with step(f"voice ({tts.VOICES[script.language]})"):
        voiceovers = make_voice(script, run_dir)
    with step("editing with ffmpeg"):
        final = ffmpeg.assemble(run_dir, script, voiceovers, args.music)

    print(f"\n{script.title!r} ({script.language})\n")
    print(summary(script, voiceovers, final))
    cost = ledger.total_cost(ledger.DEFAULT_PATH) - cost_before
    print(
        f"made in {time.perf_counter() - start:.0f} s | this run ${cost:.6f} list price "
        f"| ledger total ${ledger.total_cost(ledger.DEFAULT_PATH):.6f}"
    )


if __name__ == "__main__":
    main()
