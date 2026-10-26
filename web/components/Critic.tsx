import { fileUrl } from "@/lib/api";
import type { RunState } from "@/lib/run-state";
import type { RunDetail } from "@/lib/types";

/** Each critic round: the score against the pass mark, what the graph's edge decided, the weak scenes
 *  (reason -> fix), the frames it looked at, and what the reviser changed for the next round. */
export function Critic({ state, detail }: { state: RunState; detail?: RunDetail }) {
  const live = state.critic.length > 0;
  const reports = live ? state.critic : (detail?.critic_rounds ?? []);
  if (reports.length === 0) return null;
  return (
    <section className="panel">
      <h2>Critic</h2>
      {reports.map((r) => {
        const decision = "then" in r ? r : undefined;
        const changes = state.revisions.find((x) => x.round === r.round)?.scenes ?? [];
        return (
          <div className="critic-round" key={r.round}>
            <div className="row">
              <span className="score">{r.score}</span>
              <span className="muted small">/ 10 · round {r.round}{decision ? ` · pass at ${decision.pass_score}` : ""} · {r.model}</span>
              {decision && <span className={`badge ${decision.then === "done" ? "done" : "running"}`}>{decision.reason}</span>}
            </div>
            <p className="small">{r.summary}</p>
            {r.weak_scenes.length > 0 && (
              <ul className="weak">
                {r.weak_scenes.map((w) => (
                  <li key={w.scene}><strong>scene {w.scene}</strong> ({w.problem}): {w.reason} <span className="muted">→ {w.fix}</span></li>
                ))}
              </ul>
            )}
            {decision && decision.frame_urls.length > 0 && (
              <div className="frames" title="the frames the critic looked at">
                {decision.frame_urls.map((u, i) => <img key={u} src={fileUrl(u)} alt={`frame of scene ${decision.frames[i]}`} />)}
              </div>
            )}
            {changes.length > 0 && (
              <ul className="weak small">
                {changes.map((c) => (
                  <li key={c.scene}>revised scene {c.scene}: {c.changed ? (
                    c.narration_before !== c.narration_after
                      ? <>“{c.narration_before}” → “{c.narration_after}”</>
                      : <>new picture: {c.visual_after}</>
                  ) : "unchanged (the reviser kept it)"}</li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
    </section>
  );
}
