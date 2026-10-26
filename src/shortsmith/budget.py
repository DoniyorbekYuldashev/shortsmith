"""--budget: a hard spending limit per run, checked before every priced step (Step 7.2).

Before a graph node runs, code estimates what it will cost and stops the run when what the run
has already spent (its own ledger lines, run_id-tagged) plus that estimate would pass the limit.
The stop is clean: the node raises BudgetExceeded before it sends anything, LangGraph keeps the
checkpoint of the last finished node, and `shortsmith resume <run> --budget <more>` goes on from there.

Where an estimate comes from:
- an LLM node: for each call it makes (the writer: options + pick), the MOST expensive of the last
  10 calls of the same step on the same model in the ledger. Our own history, so it follows prompt
  changes; the most expensive, so a typical call costs less than estimated and a run stops early
  rather than late. With no history yet: a first-call guess from Phase 6's measured token counts.
- the producer: the images and the hero clip it would make right now (producer.pending), at
  ledger.MEDIA_PRICES; $0 with MOCK_MEDIA (placeholders and mock clips are free).
- approve_hero and editor: free, never stopped.

Where it can be wrong: a writer that has to write its 3 options twice (both attempts broke a
rule) makes one call more than estimated; a critic that thinks longer than usual; a model with
no history (the guess). Each can push a run over by at most that one node's error, because the
check runs again before the next node.

The budget counts LIST price. On Gemini's free tier every call is billed $0; the list price says
what the same run would cost on a paid tier, which is what a budget is meant to cap.

OpenAI's free daily tokens (data sharing, costs.py) get their own check before every LLM node: when
the node's calls would take a group past today's allowance (by our ledger's count), the run warns
(OPENAI_FREE_LIMIT_ACTION=warn, the default) or stops cleanly like a budget stop (=stop), so the credit
meant for images isn't spent on text without a word.
"""

import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from shortsmith import costs, ledger, llm
from shortsmith.agents import critic, producer
from shortsmith.agents.state import RunState
from shortsmith.media import animate, images
from shortsmith.models import CRITIC_MAX_WEAK, SCENES_PLANNED

HISTORY = 10  # past calls looked at per step and model

# (prompt, completion incl. thinking) tokens for a call with no history yet: Phase 6's averages, rounded up.
FIRST_CALL_TOKENS: dict[str, tuple[int, int]] = {
    "researcher": (1500, 350), "rag": (600, 0), "writer": (800, 2000), "writer-pick": (700, 200),
    "director": (700, 550), "critic": (4500, 1500), "revise": (750, 450),
}


def node_calls(name: str) -> list[tuple[str, str]]:
    """The (ledger step, model) of every call an LLM node makes; [] for a node without LLM calls."""
    calls = {
        "researcher": [("researcher", llm.model_for("researcher")), ("rag", llm.embedding_model())],
        "writer": [("writer", llm.model_for("writer")), ("writer-pick", llm.model_for("writer-pick"))],
        "director": [("director", llm.model_for("director"))],
        "critic": [("critic", critic.critic_model())],
        "revise": [("revise", llm.model_for("revise"))],
    }
    return calls.get(name, [])


class Estimate(BaseModel):
    """What one node is expected to cost before it runs (list price), and why."""

    node: str
    usd: float
    basis: str  # e.g. "writer $0.0057 + writer-pick $0.0007 (most expensive of the last 10)"


class BudgetExceeded(RuntimeError):
    """The next step would take the run over its budget: stopped before it, nothing sent."""

    def __init__(self, node: str, spent: float, estimate: float, limit: float):
        self.node, self.spent, self.estimate, self.limit = node, spent, estimate, limit
        super().__init__(f"over budget before {node}: spent ${spent:.4f} + {node} ~${estimate:.4f} "
                         f"> budget ${limit:.4f} (list price)")


def same_calls(step: str, model: str, history: list[ledger.LedgerEntry]) -> list[ledger.LedgerEntry]:
    """The last HISTORY calls of this step on this model (dated snapshot names match their model)."""
    base = ledger.base_model(model)
    return [e for e in history if e.step == step and ledger.base_model(e.model) == base][-HISTORY:]


def call_cost(step: str, model: str, history: list[ledger.LedgerEntry]) -> tuple[float, bool]:
    """(estimated list price of one call, True if it came from the ledger's history)."""
    past = [e.cost_usd for e in same_calls(step, model, history)]
    if past:
        return max(past), True
    prompt, completion = FIRST_CALL_TOKENS.get(step, (1000, 500))
    return ledger.cost_usd(model, prompt, completion), False


def llm_estimate(name: str, history: list[ledger.LedgerEntry]) -> Estimate:
    parts, total, guessed = [], 0.0, False
    for step, model in node_calls(name):
        usd, known = call_cost(step, model, history)
        total += usd
        guessed = guessed or not known
        parts.append(f"{step} ${usd:.4f}")
    if not parts:
        return Estimate(node=name, usd=0.0, basis="free (no LLM call)")
    source = "first-call guess" if guessed else f"most expensive of the last {HISTORY}"
    return Estimate(node=name, usd=total, basis=f"{' + '.join(parts)} ({source})")


def media_cost(todo: producer.Made) -> float:
    """List price of what a producer pass makes. Voices are free (edge-tts); mock media is free."""
    if images.mock_media():
        return 0.0
    return len(todo.images) * images.price_per_image() + len(todo.clips) * ledger.MEDIA_PRICES[animate.CLIP_MODEL]


def estimate(name: str, state: RunState, history: list[ledger.LedgerEntry]) -> Estimate:
    """What node `name` would cost if it ran now on this state."""
    if name == "producer" and state.plan is not None:
        todo = producer.pending(state.folder, state.plan, state.hero)
        usd = media_cost(todo)
        kind = "mock media, free" if images.mock_media() else f"real media: images {images.source().label}"
        return Estimate(node=name, usd=usd, basis=f"{todo} ({kind})")
    return llm_estimate(name, history)


def spent(run_id: str, history: list[ledger.LedgerEntry]) -> float:
    return sum(e.cost_usd for e in history if e.run_id == run_id)


log = logging.getLogger("shortsmith.budget")


class FreeAllowanceReached(RuntimeError):
    """OPENAI_FREE_LIMIT_ACTION=stop: the next node would pass today's free OpenAI tokens. Nothing was sent."""

    hint = ("wait for the next allowance (00:00 UTC), or let it bill: OPENAI_FREE_LIMIT_ACTION=warn")

    def __init__(self, node: str, group: str, used: int, need: int, daily: int):
        self.node, self.group, self.used, self.need, self.daily = node, group, used, need, daily
        super().__init__(f"stopped before {node}: its ~{need:,} tokens would pass today's free OpenAI tokens "
                         f"({group} group: {used:,} used of {daily:,} since 00:00 UTC, by our count)")


def call_tokens(step: str, model: str, history: list[ledger.LedgerEntry]) -> int:
    """Tokens one call is expected to use: the most of the last HISTORY such calls, else the first-call guess."""
    past = [costs.tokens_of(e) for e in same_calls(step, model, history)]
    prompt, completion = FIRST_CALL_TOKENS.get(step, (1000, 500))
    return max(past) if past else prompt + completion


def allowance_check(name: str, history: list[ledger.LedgerEntry], now: datetime | None = None) -> str | None:
    """None if node `name` fits in today's free OpenAI tokens (or data sharing is off); else a warning
    (logged), or FreeAllowanceReached with OPENAI_FREE_LIMIT_ACTION=stop."""
    settings = costs.allowance()
    if not settings.data_sharing:
        return None
    need: dict[str, int] = {}
    for step, model in node_calls(name):
        if (group := costs.free_group(model)) is not None:
            need[group] = need.get(group, 0) + call_tokens(step, model, history)
    used = costs.used_today(history, now or datetime.now(timezone.utc))
    for group, tokens in need.items():
        if used[group] + tokens > settings.daily_tokens[group]:
            if settings.action == "stop":
                raise FreeAllowanceReached(name, group, used[group], tokens, settings.daily_tokens[group])
            warning = (f"{name}: ~{tokens:,} tokens take the {group} group past today's free OpenAI tokens "
                       f"({used[group]:,} used of {settings.daily_tokens[group]:,} since 00:00 UTC, by our count): "
                       "the rest is billed at list price")
            log.warning(warning)
            return warning
    return None


def check(name: str, state: RunState, path: Path = ledger.DEFAULT_PATH) -> Estimate:
    """The node's estimate; raises BudgetExceeded if running it would take the run over its budget.
    A free node is never stopped: stopping it saves nothing (the next priced node is checked).
    Then the free-allowance check (allowance_check): a warning, or FreeAllowanceReached."""
    history = ledger.entries(path)
    found = estimate(name, state, history)
    if state.budget_usd is not None and found.usd > 0:
        so_far = spent(state.run_id, history)
        if so_far + found.usd > state.budget_usd:
            raise BudgetExceeded(name, so_far, found.usd, state.budget_usd)
    allowance_check(name, history)
    return found


class RunEstimate(BaseModel):
    """A whole run's cost before it starts (list prices): what --real shows before asking."""

    images: int
    images_usd: float
    clips: int
    clip_usd: float  # asked again before it is made (the approval step)
    revision_images: int  # at most: the critic sends up to 3 scenes back per round
    revision_usd: float
    llm_usd: float  # list price; billed $0 on the Gemini free tier or within OpenAI's free daily tokens
    image_source: str = ""  # e.g. "openai gpt-image-2 (low, 1024x1536), ~$0.0050 an image"

    @property
    def first_pass_usd(self) -> float:
        return self.images_usd + self.clip_usd + self.llm_usd

    @property
    def most_usd(self) -> float:
        return self.first_pass_usd + self.revision_usd


def run_estimate(research: bool, hero_clips: int, max_rounds: int, real: bool,
                 history: list[ledger.LedgerEntry], images_left: int = SCENES_PLANNED,
                 llm_nodes: list[str] | None = None) -> RunEstimate:
    """A new run: every image, the hero clip, and the LLM nodes of the first pass plus every round.
    A resume passes what is left (images_left, llm_nodes)."""
    if llm_nodes is None:
        llm_nodes = (["researcher"] if research else []) + ["writer", "director", "critic"] + ["revise", "critic"] * max_rounds
    chosen = images.source(real)
    image = chosen.price_usd
    clip = ledger.MEDIA_PRICES[animate.CLIP_MODEL] if real else 0.0
    rounds_images = max_rounds * CRITIC_MAX_WEAK
    return RunEstimate(images=images_left, images_usd=images_left * image, clips=hero_clips, clip_usd=hero_clips * clip,
                       revision_images=rounds_images, revision_usd=rounds_images * image,
                       llm_usd=sum(llm_estimate(node, history).usd for node in llm_nodes), image_source=chosen.label)


def nodes_left(state: RunState) -> list[str]:
    """The LLM nodes a resumed run will still run, at most (for the --real question on resume)."""
    left = [] if state.research or not state.research_on else ["researcher"]
    left += [] if state.chosen else ["writer"]
    left += [] if state.plan else ["director"]
    reviews_left = max(0, state.max_rounds + 1 - len(state.critic_rounds))
    revisions_left = max(0, state.max_rounds - len({r.round for r in state.revisions}))
    return left + ["critic"] * reviews_left + ["revise"] * revisions_left


def suggested(state: RunState, history: list[ledger.LedgerEntry]) -> float:
    """After a stop: spent so far + every LLM node still allowed, rounded up to a whole cent. A budget
    that surely finishes, not the cheapest one (the CLI's `next:` line and the web page's button)."""
    need = spent(state.run_id, history) + sum(llm_estimate(n, history).usd for n in nodes_left(state))
    return max(0.01, math.ceil(need * 100) / 100)
