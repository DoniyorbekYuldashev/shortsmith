"use client";

import Link from "next/link";
import { useCallback, useEffect, useReducer, useState } from "react";

import { ApprovalCard, BudgetCard, ErrorCard } from "@/components/Actions";
import { Costs } from "@/components/Costs";
import { Critic } from "@/components/Critic";
import { Scenes } from "@/components/Scenes";
import { PhaseBadge, StatusBadge } from "@/components/StatusBadge";
import { Timeline } from "@/components/Timeline";
import { ApiError, api, eventsUrl, fileUrl } from "@/lib/api";
import { usd } from "@/lib/format";
import { EVENT_TYPES, type RunState, initial, reduce } from "@/lib/run-state";
import type { Problem, RunDetail, RunEvent } from "@/lib/types";

type Action = { kind: "reset" } | { kind: "event"; event: RunEvent };
const step = (state: RunState, action: Action) => (action.kind === "reset" ? initial : reduce(state, action.event));

// Events after which the run's script on disk changed: fetch GET /runs/{id} again for the scenes' words.
const REFETCH = new Set(["writer", "director", "revise", "producer", "editor"]);

type Connection = "connecting" | "live" | "reconnecting" | "closed";

/** One run, live: every event from GET /runs/{id}/events folded into the page's state. A reload replays
 *  the stream from the first event, so the page comes back exactly where the run is. */
export function RunView({ id }: { id: string }) {
  const [state, dispatch] = useReducer(step, initial);
  const [detail, setDetail] = useState<RunDetail>();
  const [connection, setConnection] = useState<Connection>("connecting");
  const [problem, setProblem] = useState<Problem>();

  const refetch = useCallback(() => {
    api.run(id).then(setDetail).catch((e) => setProblem(e instanceof ApiError ? e.problem : { message: String(e), hint: "" }));
  }, [id]);

  useEffect(() => {
    dispatch({ kind: "reset" });
    refetch();
    // EventSource reconnects by itself after a dropped connection and sends Last-Event-ID; the server
    // then sends only what came after. retry: 3000 (from the stream) is how long it waits.
    const source = new EventSource(eventsUrl(id));
    const onEvent = (message: MessageEvent<string>) => {
      const event = JSON.parse(message.data) as RunEvent;
      dispatch({ kind: "event", event });
      if ((event.type === "node_finished" && REFETCH.has(event.node)) || event.type === "done") refetch();
      if (event.type === "done") {
        source.close(); // finished: no reconnecting (the server would answer 204 anyway)
        setConnection("closed");
      }
    };
    for (const type of EVENT_TYPES) source.addEventListener(type, onEvent as EventListener);
    source.onopen = () => setConnection("live");
    source.onerror = () => setConnection(source.readyState === EventSource.CLOSED ? "closed" : "reconnecting");
    return () => source.close();
  }, [id, refetch]);

  const act = async (work: () => Promise<unknown>) => {
    setProblem(undefined);
    try {
      await work();
    } catch (e) {
      setProblem(e instanceof ApiError ? e.problem : { message: String(e), hint: "" });
      throw e;
    }
  };

  const s = state.settings;
  const title = detail?.title ?? s?.topic ?? detail?.topic ?? id;
  const noEvents = state.lastId === 0 && detail !== undefined && connection !== "connecting";
  const video = state.video?.url ?? (detail?.files.final_mp4 ? `/runs/${id}/files/final.mp4` : undefined);

  return (
    <>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <h1>{title}</h1>
          <div className="muted small">
            <span className="mono">{id}</span>
            {(s?.topic ?? detail?.topic) && <> · “{s?.topic ?? detail?.topic}”</>}
            {(s?.language ?? detail?.language) && <> · {s?.language ?? detail?.language}</>}
          </div>
        </div>
        <div className="row">
          {state.lastId > 0 ? <PhaseBadge phase={state.phase} /> : detail && <StatusBadge status={detail.status} />}
          <span className="small muted">{connection === "live" ? "● live" : connection === "reconnecting" ? "reconnecting..." : connection === "closed" ? "replayed" : "connecting..."}</span>
        </div>
      </div>
      {s && (
        <p className="small muted">
          {s.research ? `research${s.brand ? ` + brand ${s.brand}` : ""} · ` : ""}up to {s.max_rounds} revision{s.max_rounds === 1 ? "" : "s"},
          pass at {s.pass_score} · hero clips {s.hero_clips} (question: {s.approve}) · {s.budget_usd !== null ? `budget ${usd(s.budget_usd)} · ` : ""}
          media {s.media} · {s.model}, critic {s.critic_model}
        </p>
      )}

      {problem && <div className="callout bad"><strong>{problem.message}</strong>{problem.hint && <p className="small">Next: {problem.hint}</p>}</div>}
      {state.phase === "queued" && state.queued && (
        <div className="callout info">
          <strong>Queued</strong>: one run at a time (the free tier is slow and Flash allows 20 requests a day).
          {state.queued.ahead.length > 0 && <> Waiting for {state.queued.ahead.map((r, i) => <span key={r}>{i > 0 && ", "}<Link href={`/runs/${r}`}>{r}</Link></span>)}.</>}
        </div>
      )}
      {state.phase === "paused" && state.approval?.waiting && !state.approval.answer && (
        <ApprovalCard approval={state.approval} onAnswer={(answer) => act(() => api.approve(id, answer))} />
      )}
      {state.phase === "over_budget" && state.budgetStop && (
        <BudgetCard stop={state.budgetStop} onResume={(budget) => act(() => api.resume(id, budget))} />
      )}
      {state.phase === "failed" && state.error && <ErrorCard error={state.error} onResume={() => act(() => api.resume(id))} />}
      {noEvents && (
        <div className="callout info small">
          This run has no live events (the CLI made it, or it is older than the web app): what you see comes from its
          checkpoint, files and the ledger. {detail?.status.kind !== "finished" && detail?.status.kind !== "no_checkpoint" && (
            <button onClick={() => act(() => api.resume(id))}>Resume it here</button>
          )}
        </div>
      )}

      {!noEvents && <Timeline state={state} />}
      <div className="grid2">
        <div>
          <Scenes runId={id} state={state} detail={detail} />
          <Critic state={state} detail={detail} />
        </div>
        <div>
          <section className="panel">
            <h2>Video</h2>
            {video ? (
              <>
                <video key={video} src={fileUrl(video)} controls playsInline preload="metadata" />
                <p className="small muted">
                  {state.phase === "done" || detail?.status.kind === "finished" ? "the final cut" : "the latest cut; the critic may send scenes back"}
                  {detail?.length_s ? ` · ${detail.length_s.toFixed(1)} s` : ""}
                </p>
              </>
            ) : <p className="muted small">appears when the editor has cut it</p>}
            {state.done && <p className="small">Critic {state.done.scores.join(" → ") || "-"} · {state.done.stop_reason}</p>}
          </section>
          <Costs state={state} detail={detail} />
        </div>
      </div>
    </>
  );
}
