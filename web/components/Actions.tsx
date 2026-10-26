"use client";

import { useState } from "react";

import { usd } from "@/lib/format";
import type { Approval } from "@/lib/run-state";
import type { EventOf } from "@/lib/types";

/** The hero-clip question while the run is paused: Yes = the AI clip, No = the director's Ken Burns move. */
export function ApprovalCard({ approval, onAnswer }: { approval: Approval; onAnswer: (answer: "yes" | "no") => Promise<void> }) {
  const [sent, setSent] = useState<"yes" | "no">();
  const answer = async (a: "yes" | "no") => {
    setSent(a);
    await onAnswer(a).catch(() => setSent(undefined));
  };
  return (
    <div className="callout warn">
      <strong>Hero clip for scene {approval.scene}?</strong>
      <p>{approval.question}</p>
      <p className="small muted">{approval.mock
        ? `Mock media: a free placeholder clip now. With real media this would cost ~${usd(approval.cost_usd, 2)}.`
        : `This costs ~${usd(approval.cost_usd, 2)}.`}</p>
      <div className="row">
        <button className="yes" disabled={sent !== undefined} onClick={() => answer("yes")}>Yes, make the clip</button>
        <button disabled={sent !== undefined} onClick={() => answer("no")}>No, use Ken Burns</button>
        {sent && <span className="small muted">sent “{sent}”: the run goes on...</span>}
      </div>
    </div>
  );
}

/** Stopped before the node that would pass the budget: nothing was sent for it; resume with more. */
export function BudgetCard({ stop, onResume }: { stop: EventOf<"budget_stopped">; onResume: (budget: number) => Promise<void> }) {
  const [budget, setBudget] = useState(String(stop.suggested_usd));
  const [busy, setBusy] = useState(false);
  const resume = async () => {
    setBusy(true);
    await onResume(Number(budget)).catch(() => setBusy(false));
  };
  return (
    <div className="callout warn">
      <strong>Stopped: over budget before the {stop.node}</strong>
      <p>Spent {usd(stop.spent_usd)} + {stop.node} ~{usd(stop.estimate_usd)} &gt; budget {usd(stop.limit_usd)} (list price;
        free tiers and free daily tokens bill $0). Everything finished is saved; nothing was sent for that step.</p>
      <div className="row">
        <input type="number" min="0" step="0.01" value={budget} onChange={(e) => setBudget(e.target.value)} style={{ width: 120 }} />
        <button className="primary" disabled={busy || !(Number(budget) > stop.spent_usd)} onClick={resume}>
          Resume with a budget of {usd(Number(budget) || 0)}
        </button>
        <span className="small muted">{usd(stop.suggested_usd)} surely finishes (every round still allowed)</span>
      </div>
    </div>
  );
}

/** A failed node (or a server restart mid-run): the CLI's one-line message, its next step, and Resume. */
export function ErrorCard({ error, onResume }: { error: EventOf<"failed">; onResume: () => Promise<void> }) {
  const [busy, setBusy] = useState(false);
  return (
    <div className="callout bad">
      <strong>{error.interrupted ? "Interrupted" : "Error"}{error.node ? ` at ${error.node}` : ""}: {error.message}</strong>
      <p className="small">Next: {error.hint}</p>
      <button disabled={busy} onClick={() => { setBusy(true); onResume().catch(() => setBusy(false)); }}>
        Resume from the last finished step
      </button>
    </div>
  );
}
