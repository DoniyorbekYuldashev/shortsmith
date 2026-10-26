import { NODE_LABEL, secs, tokens, usd } from "@/lib/format";
import { type RunState, byNode, spent } from "@/lib/run-state";
import type { RunDetail } from "@/lib/types";

/** Tokens and list price per node and for the run (Step 8.3), and the budget used if there is one. */
export function Costs({ state, detail }: { state: RunState; detail?: RunDetail }) {
  const live = state.steps.length > 0;
  const rows = live ? byNode(state) : (detail?.nodes ?? []).map((n) => ({
    node: n.node, runs: 0, calls: n.calls, promptTokens: n.prompt_tokens, completionTokens: n.completion_tokens,
    costUsd: n.cost_usd, seconds: n.seconds ?? 0,
  }));
  const total = live ? spent(state) : (detail?.cost_usd ?? 0);
  const budget = state.settings?.budget_usd ?? detail?.settings?.budget_usd ?? null;
  const freeTier = state.settings?.free_tier ?? true;
  return (
    <section className="panel">
      <h2>Tokens and cost</h2>
      <table className="small">
        <thead><tr><th>node</th><th className="num">calls</th><th className="num">in</th><th className="num">out</th><th className="num">list price</th><th className="num">time</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.node}>
              <td>{NODE_LABEL[r.node] ?? r.node}{r.runs > 1 ? <span className="muted"> ×{r.runs}</span> : null}</td>
              <td className="num">{r.calls || ""}</td>
              <td className="num">{r.calls ? tokens(r.promptTokens) : ""}</td>
              <td className="num">{r.calls ? tokens(r.completionTokens) : ""}</td>
              <td className="num">{r.calls ? usd(r.costUsd) : ""}</td>
              <td className="num">{secs(r.seconds)}</td>
            </tr>
          ))}
          <tr className="total">
            <td>total</td>
            <td className="num">{state.done?.calls ?? rows.reduce((n, r) => n + r.calls, 0)}</td>
            <td className="num" colSpan={2}>{tokens(state.done?.tokens ?? rows.reduce((n, r) => n + r.promptTokens + r.completionTokens, 0))} tokens</td>
            <td className="num">{usd(total)}</td>
            <td />
          </tr>
        </tbody>
      </table>
      <p className="small muted">
        {freeTier ? <>List price: what these calls would cost on a paid tier. <strong>Billed: $0</strong> (Gemini free tier).</>
          : <>Paid tier: list price is what is billed.</>} Media: mock placeholders, free.
      </p>
      {budget !== null && (
        <div className="small">
          Budget: {usd(total)} of {usd(budget)} used (list price)
          <div className={`meter${total > budget ? " over" : ""}`}><div style={{ width: `${Math.min(100, (total / Math.max(budget, 1e-9)) * 100)}%` }} /></div>
        </div>
      )}
    </section>
  );
}
