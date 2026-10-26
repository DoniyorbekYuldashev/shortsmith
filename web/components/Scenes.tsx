import { fileUrl } from "@/lib/api";
import type { RunState } from "@/lib/run-state";
import type { RunDetail } from "@/lib/types";

const pad = (n: number) => String(n).padStart(2, "0");

/** One card per scene: the thumbnail appears the moment the producer makes it (a media_made event). */
export function Scenes({ runId, state, detail }: { runId: string; state: RunState; detail?: RunDetail }) {
  const fromEvents = Object.keys(state.scenes).length > 0;
  const count = detail?.scenes.length || Math.max(0, ...Object.keys(state.scenes).map(Number)) || 6;
  const hero = detail?.hero;
  const revised = new Map<number, number>();
  for (const r of state.revisions) for (const c of r.scenes) if (c.changed) revised.set(c.scene, r.round + 1);
  for (const r of detail?.revisions ?? []) if (!revised.has(r.scene)) revised.set(r.scene, r.round + 1);

  return (
    <section className="panel">
      <h2>Scenes</h2>
      <div className="scenes">
        {Array.from({ length: count }, (_, i) => i + 1).map((idx) => {
          const files = state.scenes[idx];
          const row = detail?.scenes.find((s) => s.idx === idx);
          // a run made by the CLI has no events: show the files it left
          const image = files?.image ? fileUrl(files.image)
            : !fromEvents && (detail?.files.images ?? 0) >= idx ? fileUrl(`/runs/${runId}/files/scenes/scene_${pad(idx)}.png`) : undefined;
          return (
            <div className="scene" key={idx}>
              <div className={`thumb${files?.image ? " appear" : ""}`}>
                {image ? <img key={image} src={image} alt={`scene ${idx}`} /> : <span>scene {idx}</span>}
              </div>
              <div className="meta">
                <strong>{idx}</strong> {row && <span className="muted">{row.motion} · {row.words} words</span>}
                {row && <div className="muted">{row.narration.length > 90 ? `${row.narration.slice(0, 90)}...` : row.narration}</div>}
                <div className="tags">
                  {files?.voice && <span className="tag">voice</span>}
                  {(files?.clip || (hero?.scene === idx && hero.approved)) && <span className="tag hero">hero clip</span>}
                  {revised.has(idx) && <span className="tag revised">revised in loop {revised.get(idx)}</span>}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
