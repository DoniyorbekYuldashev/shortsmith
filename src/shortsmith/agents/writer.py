"""The Writer: three script options in one call, then one call to pick the best (Step 6.2).

    topic (+ research notes) -> 1 structured call -> 3 options -> code checks each -> 1 pick call -> Script

Why two calls and not four (three writes + a pick) or one (write three and pick in the same
answer)? Three options in one answer are written side by side, so the model makes them differ
(asked for different hook patterns and angles) for the price of one prompt. Picking in a separate
call means the choice is made by reading the finished options, not while still writing them.
And code sits in between: options that break a rule (too few words, a hook over 10 words) are
dropped before the pick, and the pick sees what code measured (word counts, copied hooks), so
the model judges only what code can't (which hook would stop a scroll). When only one option
survives, the pick call is skipped.

Prompts are versioned (PROMPTS): every run records which version wrote its script, so the 6.8
evaluation can compare them. v1 is the Phase 1-5 writer prompt; v2 targets Phase 5's open items.
"""

import difflib
import json
import re

from pydantic import BaseModel, ConfigDict, ValidationError

from shortsmith import llm
from shortsmith.llm import ChatResult, Message
from shortsmith.models import NARRATION_WORDS, SCENES_PLANNED, ResearchNotes, Script
from shortsmith.research import rag

OPTIONS = 3
ATTEMPTS = 2  # all options broke a rule: write once more, then give up
COPY_RATIO = 0.85  # difflib ratio of normalised text at or above this = copied, not adapted

_OPTIONS_TASK = f"""

You write {OPTIONS} DIFFERENT options for the same video. Each one is a complete script that follows \
every rule above. Make them really different: a different hook pattern and a different angle each time, \
not the same script reworded."""

_PER_SCENE = {code: (round(low / SCENES_PLANNED), round(high / SCENES_PLANNED)) for code, (low, high) in NARRATION_WORDS.items()}
_V2_TARGETS = "\n".join(
    f"  - {llm.LANGUAGES[code]}: {_PER_SCENE[code][0]}-{_PER_SCENE[code][1]} words per scene, {low}-{high} in total"
    for code, (low, high) in NARRATION_WORDS.items()
)

# v2: the same job, aimed at what Phase 5 measured going wrong: scripts off the word target (7 of
# 8), hooks copied word for word from the notes (3 of 4), narration that describes the picture
# (the coffee-shop loss), and the notes' "angles" pulled into narration as picture descriptions.
WRITER_V2 = f"""You write scripts for vertical short videos (YouTube Shorts, Reels, TikTok).
A narrator speaks over {SCENES_PLANNED} pictures, one per scene. The pictures SHOW; the narrator TELLS.

Rules:
- Exactly {SCENES_PLANNED} scenes. Count the words of every narration. Words per scene and in total:
{_V2_TARGETS}
  A script outside the total range is thrown away, so count before you answer.
- hook: the first spoken line, AT MOST 8 WORDS, written by you for THIS video. Scene 1's narration
  starts with it. Hook patterns in the notes are ideas, not sentences: never reuse an example hook.
- Narration talks TO the viewer: a fact, a tip, a reason, a feeling, a promise. Never describe what the
  picture shows ("warm lights glow", "steam rises from the cup"): the picture already shows it.
  Looks, light, colour and texture go into visual_prompt only.
- cta: one short call to action. The last scene's narration ends with the cta, so it is spoken.
- narration: spoken words only (no emojis, hashtags or stage directions).
- duration_s: your estimate of how long the scene's narration takes to say.
- visual_prompt: one vertical 9:16 image, concrete and specific to the place and topic, no text in
  the image. End every visual_prompt with the same style words, so all scenes look consistent.
- motion: zoom_in, zoom_out, pan_left or pan_right. Never ai_clip.
- Write title, hook, narration and cta in the requested language; visual_prompt always in English.
- Number scenes with idx 1, 2, 3, ... in order."""

PROMPTS = {"v1": llm.WRITER_SYSTEM_PROMPT + _OPTIONS_TASK, "v2": WRITER_V2 + _OPTIONS_TASK}
DEFAULT_VERSION = "v2"

PICK_SYSTEM_PROMPT = """You pick the best of a few script options for a ~30-second vertical short video.
Score every option from 1 to 10 on:
- hook: would the first spoken line stop a viewer from scrolling?
- clarity: one clear idea from hook to payoff to call to action, in natural spoken language.
- fit: does it suit the topic, the viewer, and the notes or brand when there are any?
Code has measured each option: its word count against the target and whether its hook was copied.
Prefer an option inside the word target with its own hook. Then give the best option's number and
say why in one sentence."""


# The shape asked for; validated option by option, not as a whole (see write_options). No docstring:
# Pydantic would send it to the model inside the JSON schema.
class ScriptOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    options: list[Script]


class OptionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option: int  # 1-based, as numbered in the message
    hook: int
    clarity: int
    fit: int


class WriterPick(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[OptionScore]
    best: int  # 1-based
    reason: str


class WriterResult(BaseModel):
    """Everything one Writer step produced, for the state and the build notes."""

    version: str
    options: list[Script]  # the options that passed the rules, in the model's order
    rejected: list[str]  # why the others were dropped
    pick: WriterPick | None  # None = only one option passed: no pick call
    chosen: Script
    results: list[ChatResult]  # every call, for usage accounting


# --- what the writer reads ---------------------------------------------------------------------


def notes_text(notes: ResearchNotes, version: str) -> str:
    """The notes as the writer reads them. v1: ResearchNotes.for_writer(), as in Phase 5.
    v2: the hook patterns without ready-made sentences (Phase 5: 3 of 4 hooks were copied from
    them), and the angles split into what the narrator SAYS and what the picture SHOWS."""
    if version == "v1":
        return notes.for_writer()
    lines = ["Research notes (real YouTube trends of the last 30 days, and proven hook patterns):", "What gets views now:"]
    lines += [f"- {item}" for item in notes.trending]
    lines.append("Hook patterns that fit (use the idea; write your own sentence for this video):")
    lines += [f"- {h.pattern}: {h.why}" for h in notes.hooks]
    lines.append("Angles. The narrator SAYS the substance (a fact, tip or reason for the viewer); "
                 "the look (light, colour, steam, texture) goes into visual_prompt only:")
    lines += [f"- {item}" for item in notes.angles]
    lines.append("Avoid:")
    lines += [f"- {item}" for item in notes.avoid]
    if notes.visual_style.strip():
        lines.append(f"End every visual_prompt with these style words: {notes.visual_style.strip()}")
    return "\n".join(lines)


def options_messages(topic: str, language: str, notes: ResearchNotes | None, version: str) -> list[Message]:
    """System = the versioned rules (the same for every topic); user = this topic and its notes."""
    user = f"Topic: {topic}\nLanguage: {llm.LANGUAGES[language]} (language code '{language}')"
    if notes:
        user += f"\n\n{notes_text(notes, version)}"
    return [{"role": "system", "content": PROMPTS[version]}, {"role": "user", "content": user}]


# --- checks code can make ------------------------------------------------------------------------


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def copied_from(hook: str, sources: dict[str, str]) -> str | None:
    """The source this hook copies (nearly) word for word, or None. `sources`: name -> text."""
    mine = _normalise(hook)
    for name, text in sources.items():
        if difflib.SequenceMatcher(None, mine, _normalise(text)).ratio() >= COPY_RATIO:
            return name
    return None


def hook_sources(notes: ResearchNotes | None) -> dict[str, str]:
    """Ready-made hooks a writer could copy: the notes' suggestions and the library's examples."""
    sources = {f"notes: {h.pattern}": h.hook for h in (notes.hooks if notes else [])}
    return sources | library_examples()


_EXAMPLE = re.compile(r'^- (en|uz|ru): "(.+)"\s*$', re.MULTILINE)
_library: dict[str, str] | None = None


def library_examples() -> dict[str, str]:
    """Every example hook in data/hooks (~200: 69 patterns x 3 languages). Read once, offline."""
    global _library
    if _library is None:
        _library = {}
        for chunk in rag.load_chunks(rag.DATA_DIR):
            if chunk.kind == "hook":
                for language, text in _EXAMPLE.findall(chunk.text):
                    _library[f"library: {chunk.title} ({language})"] = text
    return _library


def words(script: Script) -> int:
    return sum(len(scene.narration.split()) for scene in script.scenes)


def option_facts(script: Script, sources: dict[str, str]) -> str:
    low, high = NARRATION_WORDS[script.language]
    n = words(script)
    where = "in range" if low <= n <= high else ("too short" if n < low else "too long")
    copied = copied_from(script.hook, sources)
    return f"words: {n}, target {low}-{high}: {where}; hook copied: {copied or 'no'}"


# --- the two calls ---------------------------------------------------------------------------------


def write_options(topic: str, language: str, notes: ResearchNotes | None, version: str,
                  model: str | None = None) -> tuple[list[Script], list[str], ChatResult]:
    """One call for OPTIONS scripts; each is validated on its own (Script's rules + the language)."""
    result = llm.paced(lambda: llm.chat_schema(options_messages(topic, language, notes, version), ScriptOptions,
                                                 model=model, step="writer"))
    valid, rejected = [], []
    try:
        raw = json.loads(result.text).get("options", [])  # the shape only: no rule checked yet
    except (json.JSONDecodeError, AttributeError):
        raw, rejected = [], ["the answer was not a JSON object with options"]
    for n, option in enumerate(raw[:OPTIONS], start=1):
        try:
            script = Script.model_validate(option)
        except ValidationError as e:
            rejected.append(f"option {n}: {'; '.join(err['msg'] for err in e.errors(include_url=False))}")
            continue
        if script.language != language:
            rejected.append(f"option {n}: written in '{script.language}', not '{language}'")
            continue
        valid.append(script)
    return valid, rejected, result


def pick_messages(topic: str, options: list[Script], sources: dict[str, str]) -> list[Message]:
    """The options' spoken words only (no visual prompts: the pick is about what is said) + facts."""
    blocks = []
    for n, script in enumerate(options, start=1):
        lines = [f"Option {n} ({option_facts(script, sources)})", f"Hook: {script.hook}"]
        lines += [f"Scene {s.idx}: {s.narration}" for s in script.scenes]
        lines.append(f"Call to action: {script.cta}")
        blocks.append("\n".join(lines))
    language = llm.LANGUAGES[options[0].language]
    user = f"Topic: {topic}\nLanguage: {language}\n\n" + "\n\n".join(blocks)
    return [{"role": "system", "content": PICK_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def pick(topic: str, options: list[Script], sources: dict[str, str],
         model: str | None = None) -> tuple[WriterPick, ChatResult]:
    choice, result = llm.paced(lambda: llm.chat_json(pick_messages(topic, options, sources), WriterPick,
                                                     model=model, step="writer-pick"))
    if not 1 <= choice.best <= len(options):  # a number off the list: keep the first, say so
        choice = choice.model_copy(update={"best": 1, "reason": f"(asked for option {choice.best}, which doesn't exist) "
                                                                  + choice.reason})
    return choice, result


def write(topic: str, language: str = "en", notes: ResearchNotes | None = None,
          version: str = DEFAULT_VERSION, model: str | None = None) -> WriterResult:
    """Options, checks, pick: the Writer node's whole job."""
    if version not in PROMPTS:
        raise ValueError(f"unknown writer prompt {version!r}; the versions are: {', '.join(PROMPTS)}")
    results: list[ChatResult] = []
    rejected: list[str] = []
    options: list[Script] = []
    for attempt in range(1, ATTEMPTS + 1):
        options, dropped, result = write_options(topic, language, notes, version, model)
        results.append(result)
        rejected += [f"attempt {attempt}, {reason}" for reason in dropped]
        if options:
            break
    if not options:
        raise ValueError(f"every script option broke a rule, {ATTEMPTS} times: {'; '.join(rejected)}")
    choice = None
    chosen = options[0]
    if len(options) > 1:
        choice, result = pick(topic, options, hook_sources(notes), model)
        results.append(result)
        chosen = options[choice.best - 1]
    return WriterResult(version=version, options=options, rejected=rejected, pick=choice, chosen=chosen, results=results)
