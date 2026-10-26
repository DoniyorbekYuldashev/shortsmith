// The live run page's state, folded from the run's events: state = events.reduce(reduce, initial).
//
// A pure function (no React, no fetch), so the same code works for a run happening now, a reload
// mid-run (the stream replays from event 1, and the page rebuilds the same state), and a finished
// run opened days later. `node --test lib/run-state.test.ts` tests it without a browser.
import type { EventOf, EventType, RunEvent } from "./types";

// Every `event:` name the stream sends: EventSource calls a listener per name (onmessage gets only unnamed
// events). The check below fails to compile if the backend adds a type this list doesn't have.
export const EVENT_TYPES = [
  "queued", "run_started", "node_started", "node_finished", "media_made", "approval_needed", "approval_answered",
  "video_made", "critic_report", "revision", "budget_stopped", "failed", "done",
] as const satisfies readonly EventType[];
type Missing = Exclude<EventType, (typeof EVENT_TYPES)[number]>;
export const ALL_EVENT_TYPES_LISTED: [Missing] extends [never] ? true : Missing = true;

// The graph's nodes in order (agents/graph.py): the first pass, then each critic -> revise loop.
export const FIRST_PASS = ["researcher", "writer", "director", "approve_hero", "producer", "editor", "critic"];
export const LOOP = ["revise", "approve_hero", "producer", "editor", "critic"];

export type StepStatus = "running" | "waiting" | "done" | "failed";

export interface Step {
  node: string;
  round: number;
  status: StepStatus;
  startedAt: string;
  doing: string;
  estimateUsd: number;
  seconds: number;
  calls: number;
  promptTokens: number;
  completionTokens: number;
  costUsd: number;
  summary: string;
}

export interface SceneFiles {
  image?: string;
  voice?: string;
  clip?: string;
  round: number; // the round that last made one of its files
}

export type Phase = "connecting" | "queued" | "running" | "paused" | "over_budget" | "failed" | "done";

export type Approval = EventOf<"approval_needed"> & { answer?: "yes" | "no"; by?: "person" | "policy" };

export interface RunState {
  lastId: number; // the last event folded in: a replayed one with an id <= this is ignored
  phase: Phase;
  settings?: EventOf<"run_started">; // the latest start or resume
  spentBefore: number; // list price spent before the first job in these events (a run the CLI started)
  queued?: EventOf<"queued">;
  steps: Step[];
  scenes: Record<number, SceneFiles>;
  approval?: Approval;
  critic: EventOf<"critic_report">[];
  revisions: EventOf<"revision">[];
  video?: EventOf<"video_made">;
  budgetStop?: EventOf<"budget_stopped">;
  error?: EventOf<"failed">;
  done?: EventOf<"done">;
}

export const initial: RunState = { lastId: 0, phase: "connecting", spentBefore: 0, steps: [], scenes: {}, critic: [], revisions: [] };

function replaceLast(steps: Step[], match: (s: Step) => boolean, change: (s: Step) => Step): Step[] | null {
  for (let i = steps.length - 1; i >= 0; i--) {
    if (match(steps[i])) return [...steps.slice(0, i), change(steps[i]), ...steps.slice(i + 1)];
  }
  return null;
}

const open = (s: Step) => s.status === "running" || s.status === "waiting";

export function reduce(state: RunState, event: RunEvent): RunState {
  if (event.id <= state.lastId) return state; // already folded in (a reconnect that overlapped)
  const s: RunState = { ...state, lastId: event.id };
  switch (event.type) {
    case "queued":
      return { ...s, phase: "queued", queued: event };
    case "run_started":
      return {
        ...s,
        phase: "running",
        settings: event,
        queued: undefined,
        budgetStop: undefined, // a resume after a stop or an error: that problem is behind it
        error: undefined,
        spentBefore: state.settings ? state.spentBefore : event.spent_usd,
      };
    case "node_started": {
      const step: Step = {
        node: event.node, round: event.round, status: "running", startedAt: event.at, doing: event.doing,
        estimateUsd: event.estimate_usd, seconds: 0, calls: 0, promptTokens: 0, completionTokens: 0, costUsd: 0, summary: "",
      };
      // approve_hero starts again after its answer (an interrupt re-runs its node): one step, not two
      const again = replaceLast(s.steps, (x) => open(x) && x.node === event.node && x.round === event.round, () => step);
      return { ...s, steps: again ?? [...s.steps, step] };
    }
    case "node_finished": {
      const finish = (x: Step): Step => ({
        ...x, status: "done", seconds: event.seconds, calls: event.calls, promptTokens: event.prompt_tokens,
        completionTokens: event.completion_tokens, costUsd: event.cost_usd, summary: event.summary,
      });
      const steps = replaceLast(s.steps, (x) => open(x) && x.node === event.node, finish);
      const blank: Step = { ...initialStep(event.node, event.round, event.at) };
      return { ...s, steps: steps ?? [...s.steps, finish(blank)] };
    }
    case "media_made": {
      const before = s.scenes[event.scene] ?? { round: event.round };
      return { ...s, scenes: { ...s.scenes, [event.scene]: { ...before, [event.kind]: event.url, round: event.round } } };
    }
    case "approval_needed": {
      if (!event.waiting) return { ...s, approval: event };
      const steps = replaceLast(s.steps, (x) => x.status === "running" && x.node === "approve_hero", (x) => ({ ...x, status: "waiting" }));
      return { ...s, phase: "paused", approval: event, steps: steps ?? s.steps };
    }
    case "approval_answered":
      return s.approval ? { ...s, approval: { ...s.approval, answer: event.answer, by: event.by } } : s;
    case "video_made":
      return { ...s, video: event };
    case "critic_report":
      return { ...s, critic: [...s.critic, event] };
    case "revision":
      return { ...s, revisions: [...s.revisions, event] };
    case "budget_stopped":
      return { ...s, phase: "over_budget", budgetStop: event };
    case "failed": {
      const steps = replaceLast(s.steps, open, (x) => ({ ...x, status: "failed" }));
      return { ...s, phase: "failed", error: event, steps: steps ?? s.steps };
    }
    case "done":
      return { ...s, phase: "done", done: event };
    default: {
      const unknown: never = event; // a new event type in the backend must be handled here (a type error otherwise)
      return unknown;
    }
  }
}

function initialStep(node: string, round: number, at: string): Step {
  return { node, round, status: "running", startedAt: at, doing: "", estimateUsd: 0, seconds: 0, calls: 0,
           promptTokens: 0, completionTokens: 0, costUsd: 0, summary: "" };
}

export const fold = (events: RunEvent[], from: RunState = initial): RunState => events.reduce(reduce, from);

// --- what the page shows, derived ---------------------------------------------------------------------

export interface RoundView {
  round: number;
  nodes: { node: string; step?: Step }[];
}

/** The timeline: one row per round (the first pass, then each critic loop), every node in its slot.
 *  A critic that sent scenes back opens the next round's row at once (all pending). */
export function rounds(state: RunState): RoundView[] {
  const research = state.settings?.research ?? state.steps.some((x) => x.node === "researcher");
  let last = Math.max(0, ...state.steps.map((x) => x.round));
  const critic = state.critic.at(-1);
  if (critic && critic.then === "revise" && state.phase !== "done") last = Math.max(last, critic.round + 1);
  const views: RoundView[] = [];
  for (let round = 0; round <= last; round++) {
    const template = round === 0 ? FIRST_PASS.filter((n) => research || n !== "researcher") : LOOP;
    views.push({
      round,
      nodes: template.map((node) => ({ node, step: state.steps.findLast((x) => x.node === node && x.round === round) })),
    });
  }
  return views;
}

export interface NodeTotal {
  node: string;
  runs: number;
  calls: number;
  promptTokens: number;
  completionTokens: number;
  costUsd: number;
  seconds: number;
}

/** Per node, summed over its runs (the critic and the producer run once per round). */
export function byNode(state: RunState): NodeTotal[] {
  const rows = new Map<string, NodeTotal>();
  for (const x of state.steps) {
    const row = rows.get(x.node) ?? { node: x.node, runs: 0, calls: 0, promptTokens: 0, completionTokens: 0, costUsd: 0, seconds: 0 };
    rows.set(x.node, {
      ...row, runs: row.runs + 1, calls: row.calls + x.calls, promptTokens: row.promptTokens + x.promptTokens,
      completionTokens: row.completionTokens + x.completionTokens, costUsd: row.costUsd + x.costUsd, seconds: row.seconds + x.seconds,
    });
  }
  const order = [...FIRST_PASS, "revise"];
  return [...rows.values()].sort((a, b) => order.indexOf(a.node) - order.indexOf(b.node));
}

/** The run's list price so far: what earlier jobs spent (a CLI-started run) + every finished node here. */
export function spent(state: RunState): number {
  return state.done ? state.done.cost_usd : state.spentBefore + state.steps.reduce((sum, x) => sum + x.costUsd, 0);
}

export const running = (state: RunState): Step | undefined => state.steps.findLast((x) => x.status === "running");
