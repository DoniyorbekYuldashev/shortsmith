"""RunState: the graph's state, one per run, checkpointed after every node (PLAN §5, adapted).

A node gets the whole state and returns only the fields it changes (a dict). How a returned
value meets the old one is the field's *reducer*:
- no reducer (most fields): the new value replaces the old one. The Writer returns `chosen`,
  and `chosen` is now that script.
- `critic_rounds`, `revisions` and `made`: operator.add, so a node returns [this round's report] and it
  is APPENDED. Without it every critic round would overwrite the last one, and the routing rule
  "at most 2 rounds" (len(critic_rounds)) and the before/after history would be lost.
- `usage`: add_usage, a custom reducer that sums tokens, cost and time per node. A node returns
  only its own numbers ({"critic": Usage(...)}); a node that runs again (critic, producer, editor
  in later rounds) adds to its own line. No node ever reads, adds and writes back a total, which
  is the kind of read-modify-write that silently loses updates once two nodes run in parallel.
"""

import operator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel

from shortsmith.agents.director import HeroPlan
from shortsmith.agents.reviser import Revision
from shortsmith.agents.writer import WriterPick
from shortsmith.llm import ChatResult
from shortsmith.models import CriticReport, ResearchNotes, Script


class Usage(BaseModel):
    """What one node consumed: its LLM calls (from the ledger's own numbers) and its wall time."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0  # visible output + thinking (both billed as output)
    cost_usd: float = 0.0  # list price
    seconds: float = 0.0  # wall time of the node: LLM waits, ffmpeg, edge-tts

    @classmethod
    def of(cls, results: list[ChatResult]) -> "Usage":
        return cls(calls=len(results), prompt_tokens=sum(r.prompt_tokens for r in results),
                   completion_tokens=sum(r.completion_tokens + r.reasoning_tokens for r in results),
                   cost_usd=sum(r.cost_usd for r in results))

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(calls=self.calls + other.calls, prompt_tokens=self.prompt_tokens + other.prompt_tokens,
                     completion_tokens=self.completion_tokens + other.completion_tokens,
                     cost_usd=self.cost_usd + other.cost_usd, seconds=round(self.seconds + other.seconds, 2))


def add_usage(old: dict[str, Usage], new: dict[str, Usage]) -> dict[str, Usage]:
    """Reducer for RunState.usage: merge per node, summing a node's numbers when it runs again."""
    merged = dict(old)
    for node, usage in new.items():
        merged[node] = merged.get(node, Usage()) + Usage.model_validate(usage)
    return merged


class RunState(BaseModel):
    # --- what the run is (set once, at the start) ---
    run_id: str  # = the run folder's name = the checkpointer's thread_id
    run_dir: str
    topic: str
    language: Literal["en", "uz", "ru"] = "en"
    brand: str | None = None
    research_on: bool = False
    prompt_version: str = "v2"  # which writer prompt (writer.PROMPTS) made this run's script
    max_rounds: int = 2  # revision rounds after the first review, at most
    pass_score: float = 7.0  # the critic's score at which a video is done
    via_mcp: bool = False  # the producer reaches the media tools through the MCP server
    # Phase 7 (the CLI). Saved with the run, so a resume keeps them unless its own flags say otherwise.
    budget_usd: float | None = None  # hard limit in LIST price; checked before every priced node (budget.py)
    hero_clips: int = 1  # 0 = Ken Burns for every scene, 1 = the director's one approval-gated hero clip
    media: Literal["mock", "real"] = "mock"  # what MOCK_MEDIA was when the run started (runs before Phase 7: mock)

    # --- what the nodes produce (replaced) ---
    research: ResearchNotes | None = None
    script_options: list[Script] = []
    rejected_options: list[str] = []
    pick: WriterPick | None = None
    chosen: Script | None = None  # the writer's pick
    plan: Script | None = None  # the director's final shots, revised by later rounds: what gets made
    hero: HeroPlan | None = None
    final_path: str | None = None

    # --- history (appended) and accounting (summed) ---
    critic_rounds: Annotated[list[CriticReport], operator.add] = []
    revisions: Annotated[list[Revision], operator.add] = []
    made: Annotated[list[str], operator.add] = []  # each producer pass, e.g. "round 1: images 3; voices 3, 5"
    usage: Annotated[dict[str, Usage], add_usage] = {}

    @property
    def folder(self) -> Path:
        return Path(self.run_dir)

    @property
    def round(self) -> int:
        """The current round: 0 until the first review, then 1 (making the first revision), 2, ..."""
        return len(self.critic_rounds)

    @property
    def tokens_used(self) -> int:  # PLAN §5's two totals, now derived from the per-node numbers
        return sum(u.prompt_tokens + u.completion_tokens for u in self.usage.values())

    @property
    def cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usage.values())
