"use client";

import { useEffect, useState } from "react";

import { NODE_LABEL, secs, tokens, usd } from "@/lib/format";
import { type RunState, type Step, rounds, running } from "@/lib/run-state";

function useNow(ticking: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!ticking) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [ticking]);
  return now;
}

const elapsed = (step: Step, now: number) => Math.max(0, (now - Date.parse(step.startedAt)) / 1000);

/** researcher -> writer -> director -> approval -> producer -> editor -> critic, then one row per critic loop. */
export function Timeline({ state }: { state: RunState }) {
  const current = running(state);
  const now = useNow(current !== undefined);
  return (
    <section className="panel">
      <h2>Pipeline</h2>
      {rounds(state).map((r) => (
        <div className="round" key={r.round}>
          <div className="round-label">{r.round === 0 ? "first pass" : `critic loop ${r.round}`}</div>
          <div className="nodes">
            {r.nodes.map(({ node, step }) => (
              <NodeCard key={node} node={node} step={step} now={now} answered={state.approval?.answer} />
            ))}
          </div>
        </div>
      ))}
      {current && (
        <p className="doing">
          <strong>{NODE_LABEL[current.node] ?? current.node}</strong>: {current.doing}{" "}
          <span className="muted">{secs(elapsed(current, now))}
            {current.estimateUsd > 0 && ` · est. ${usd(current.estimateUsd)} list price`}
            {current.node === "critic" && " · a free-tier critic call can wait minutes in Google's queue"}</span>
        </p>
      )}
    </section>
  );
}

function NodeCard({ node, step, now, answered }: { node: string; step?: Step; now: number; answered?: "yes" | "no" }) {
  const status = step?.status ?? "pending";
  return (
    <div className={`node ${status}`} title={step?.summary || undefined}>
      <div className="name"><span className="dot" />{NODE_LABEL[node] ?? node}</div>
      <div className="facts">
        {!step && "waiting"}
        {step?.status === "running" && <>running {secs(elapsed(step, now))}</>}
        {step?.status === "waiting" && (answered ? `you said ${answered}: goes on when the worker is free` : "waiting for your answer")}
        {step?.status === "failed" && "failed"}
        {step?.status === "done" && (
          <>
            {secs(step.seconds)}
            {step.calls > 0 && (
              <>
                <br />{tokens(step.promptTokens)} in · {tokens(step.completionTokens)} out
                <br />{usd(step.costUsd)}
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
