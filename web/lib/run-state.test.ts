// The reducer, with Node's own test runner (no test library to install): `npm test` in web/.
// Node 24 runs TypeScript by stripping the types (erasableSyntaxOnly in tsconfig keeps the code strippable).
import assert from "node:assert/strict";
import { test } from "node:test";

import { byNode, fold, initial, reduce, rounds, spent } from "./run-state.ts";
import type { EventOf, EventType, RunEvent } from "./types";

let id = 0;
const at = "2026-10-25T10:00:00Z";

function ev<T extends EventType>(type: T, fields: Omit<EventOf<T>, "id" | "at" | "type">): EventOf<T> {
  return { id: ++id, at, type, ...fields } as EventOf<T>;
}

const started = (resumed = false, spent_usd = 0) => ev("run_started", {
  resumed, next: resumed ? ["critic"] : [], topic: "coffee", language: "en", research: false, brand: null,
  max_rounds: 2, pass_score: 7, hero_clips: 1, approve: "ask", budget_usd: 0.01, spent_usd, media: "mock",
  provider: "gemini", model: "gemini-3.5-flash-lite", critic_model: "gemini-3.8-flash", free_tier: true,
});
const begin = (node: string, round = 0) => ev("node_started", { node, round, doing: `${node}...`, estimate_usd: 0.001, basis: "x" });
const end = (node: string, round = 0, cost = 0.001, calls = 1) => ev("node_finished", {
  node, round, seconds: 1.5, calls, prompt_tokens: 100 * calls, completion_tokens: 50 * calls, cost_usd: cost, summary: `${node} done`,
});
const question = (waiting: boolean) => ev("approval_needed", {
  question: "Scene 1 as a clip costs ~$0.30. Continue?", scene: 1, motion: "steam rises", cost_usd: 0.3, mock: true, waiting,
});
const report = (round: number, score: number, then: "revise" | "done") => ev("critic_report", {
  round, score, pass_score: 7, summary: "s", hook: "h", weak_scenes: then === "revise" ? [{ scene: 3, problem: "visual", reason: "r", fix: "f" }] : [],
  frames: [1, 3], frame_urls: [], model: "gemini-3.8-flash", then, reason: then === "done" ? "passed" : "revising scene(s) 3",
});

function firstPass(): RunEvent[] {
  return [
    ev("queued", { position: 1, ahead: [] }), started(),
    begin("writer"), end("writer", 0, 0.004, 2), begin("director"), end("director"),
    begin("approve_hero"), question(false), ev("approval_answered", { answer: "yes", by: "policy" }), begin("approve_hero"), end("approve_hero", 0, 0, 0),
    begin("producer"), ev("media_made", { kind: "image", scene: 1, round: 0, path: "scenes/scene_01.png", url: "/runs/r/files/scenes/scene_01.png?v=0" }),
    end("producer", 0, 0, 0), begin("editor"), end("editor", 0, 0, 0), ev("video_made", { round: 0, url: "/runs/r/files/final.mp4?v=0" }),
    begin("critic"), end("critic", 0, 0.006), report(0, 5, "revise"),
  ];
}

test("a first pass folds into steps, scenes, the video and a critic round", () => {
  const s = fold(firstPass());
  assert.equal(s.phase, "running");
  assert.deepEqual(s.steps.map((x) => x.node), ["writer", "director", "approve_hero", "producer", "editor", "critic"]);
  assert.ok(s.steps.every((x) => x.status === "done"));
  assert.equal(s.steps[0].calls, 2);
  assert.equal(s.scenes[1].image, "/runs/r/files/scenes/scene_01.png?v=0");
  assert.equal(s.video?.url, "/runs/r/files/final.mp4?v=0");
  assert.equal(s.approval?.answer, "yes");
  assert.equal(spent(s), 0.004 + 0.001 + 0.006);
});

test("approve_hero starting again after its answer is one step, not two", () => {
  const s = fold(firstPass());
  assert.equal(s.steps.filter((x) => x.node === "approve_hero").length, 1);
});

test("a critic that sends scenes back opens the next round's row, all pending", () => {
  const view = rounds(fold(firstPass()));
  assert.equal(view.length, 2);
  assert.deepEqual(view[0].nodes.map((n) => n.node), ["writer", "director", "approve_hero", "producer", "editor", "critic"]);
  assert.deepEqual(view[1].nodes.map((n) => n.node), ["revise", "approve_hero", "producer", "editor", "critic"]);
  assert.ok(view[1].nodes.every((n) => n.step === undefined));
});

test("the loop round fills in, and done ends it with the ledger's numbers", () => {
  const s = fold([...firstPass(), begin("revise", 1), end("revise", 1, 0.002), begin("critic", 1), end("critic", 1, 0.006),
    report(1, 8, "done"), ev("done", { final_url: "/f", scores: [5, 8], stop_reason: "passed (8 >= 7)", seconds: 60,
      calls: 7, tokens: 1000, cost_usd: 0.02, billed_usd: 0, budget_usd: 0.05 })]);
  assert.equal(s.phase, "done");
  assert.equal(rounds(s)[1].nodes[0].step?.status, "done");
  const critic = byNode(s).find((n) => n.node === "critic");
  assert.equal(critic?.runs, 2);
  assert.equal(critic?.calls, 2);
  assert.equal(spent(s), 0.02); // the done event's total (from the ledger) wins over the sum
});

test("a waiting question pauses the run; the answer and the resume go on", () => {
  let s = fold([ev("queued", { position: 1, ahead: [] }), started(), begin("approve_hero"), question(true)]);
  assert.equal(s.phase, "paused");
  assert.equal(s.steps[0].status, "waiting");
  s = fold([ev("approval_answered", { answer: "no", by: "person" }), ev("queued", { position: 1, ahead: [] }), started(true),
    begin("approve_hero"), end("approve_hero", 0, 0, 0)], s);
  assert.equal(s.phase, "running");
  assert.equal(s.approval?.answer, "no");
  assert.equal(s.steps.length, 1);
  assert.equal(s.steps[0].status, "done");
});

test("a budget stop, then a resume clears it", () => {
  let s = fold([started(), begin("writer"), end("writer", 0, 0.007), ev("budget_stopped", {
    node: "critic", spent_usd: 0.007, estimate_usd: 0.0074, limit_usd: 0.01, suggested_usd: 0.04, message: "over budget before critic" })]);
  assert.equal(s.phase, "over_budget");
  assert.equal(s.budgetStop?.suggested_usd, 0.04);
  s = reduce(s, started(true, 0.007));
  assert.equal(s.phase, "running");
  assert.equal(s.budgetStop, undefined);
  assert.equal(s.spentBefore, 0); // the first start's: the steps already hold what was spent
});

test("an error marks the open step failed", () => {
  const s = fold([started(), begin("director"), ev("failed", { node: "director", message: "boom", hint: "resume", interrupted: false })]);
  assert.equal(s.phase, "failed");
  assert.equal(s.steps[0].status, "failed");
});

test("a replayed event (an id already folded in) changes nothing", () => {
  const events = firstPass();
  const s = fold(events);
  assert.equal(fold(events.slice(5), s), s);
  assert.equal(reduce(s, events[0]), s);
});

test("a run the CLI started counts what it spent before the page's first event", () => {
  const s = fold([started(true, 0.01), begin("critic"), end("critic", 0, 0.006)], initial);
  assert.equal(spent(s), 0.016);
});
