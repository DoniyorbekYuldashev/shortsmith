"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { type FormEvent, useEffect, useState } from "react";

import { StatusBadge } from "@/components/StatusBadge";
import { ApiError, api } from "@/lib/api";
import { usd, when } from "@/lib/format";
import type { Health, Options, Problem, RunSummary, StartRun } from "@/lib/types";

// Every field is sent, defaults included: the generated StartRun type has them all as required (openapi-typescript
// types a field with a default as always present, which is right for responses and strict for requests).
type Form = Omit<StartRun, "budget_usd">;

const START: Form = { topic: "", language: "en", research: false, brand: null, approve: "ask", max_rounds: 2, pass_score: 7, hero_clips: 1 };

export default function Home() {
  const router = useRouter();
  const [form, setForm] = useState<Form>(START);
  const [budget, setBudget] = useState("");
  const [options, setOptions] = useState<Options>();
  const [health, setHealth] = useState<Health>();
  const [recent, setRecent] = useState<RunSummary[]>();
  const [problem, setProblem] = useState<Problem>();
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const failed = (e: unknown) => setProblem(e instanceof ApiError ? e.problem : { message: String(e), hint: "" });
    api.options().then(setOptions).catch(failed);
    api.runs(6).then(setRecent).catch(failed);
    api.health().then(setHealth).catch(failed); // slower: it asks Docker and the database
  }, []);

  const set = <K extends keyof Form>(key: K, value: Form[K]) => setForm((f) => ({ ...f, [key]: value }));

  async function start(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setProblem(undefined);
    try {
      const accepted = await api.start({ ...form, budget_usd: budget.trim() ? Number(budget) : null });
      router.push(`/runs/${accepted.run_id}`);
    } catch (err) {
      setProblem(err instanceof ApiError ? err.problem : { message: String(err), hint: "" });
      setBusy(false);
    }
  }

  const researchDown = health !== undefined && !health.research_ready;

  return (
    <>
      <h1>Make a short video</h1>
      <p className="muted">One line of text in: research (optional), a writer, a director, the hero-clip question, the producer,
        the editor and a critic loop. You watch every step live.</p>

      {problem && (
        <div className="callout bad">
          <strong>{problem.message}</strong>
          {problem.hint && <p className="small">Next: {problem.hint}</p>}
        </div>
      )}

      <div className="grid2">
        <form className="panel stack" onSubmit={start}>
          <div>
            <label htmlFor="topic">Topic</label>
            <input id="topic" type="text" required maxLength={200} autoFocus placeholder="coffee shop in Tashkent, cozy autumn vibe"
                   value={form.topic} onChange={(e) => set("topic", e.target.value)} />
          </div>
          <div className="fields">
            <div>
              <label htmlFor="language">Language</label>
              <select id="language" value={form.language} onChange={(e) => set("language", e.target.value as Form["language"])}>
                <option value="en">English</option>
                <option value="uz">Uzbek</option>
                <option value="ru">Russian</option>
              </select>
            </div>
            <div>
              <label htmlFor="brand">Brand</label>
              <select id="brand" value={form.brand ?? ""} disabled={researchDown}
                      onChange={(e) => set("brand", e.target.value || null)}>
                <option value="">none</option>
                {options?.brands.map((b) => <option key={b} value={b}>{b}</option>)}
              </select>
            </div>
            <div>
              <label htmlFor="approve">Hero clip</label>
              <select id="approve" value={form.approve} onChange={(e) => set("approve", e.target.value as Form["approve"])}>
                <option value="ask">ask me (the run pauses)</option>
                <option value="yes">yes, make it</option>
                <option value="no">no, Ken Burns</option>
              </select>
            </div>
            <div>
              <label htmlFor="rounds">Critic rounds</label>
              <select id="rounds" value={form.max_rounds} onChange={(e) => set("max_rounds", Number(e.target.value))}>
                {[0, 1, 2, 3].map((n) => <option key={n} value={n}>{n === 0 ? "review once" : `up to ${n} revision${n > 1 ? "s" : ""}`}</option>)}
              </select>
            </div>
            <div>
              <label htmlFor="budget">Budget (USD, list price)</label>
              <input id="budget" type="number" min="0" step="0.001" placeholder="none" value={budget}
                     onChange={(e) => setBudget(e.target.value)} />
            </div>
          </div>
          <label className="check">
            <input type="checkbox" checked={form.research || form.brand !== null} disabled={researchDown || form.brand !== null}
                   onChange={(e) => set("research", e.target.checked)} />
            Research: this month&apos;s YouTube trends + the hook library (1 more LLM call)
          </label>
          {researchDown && <p className="small muted">Research is off: {health?.problems.find((p) => p.message.startsWith("research"))?.hint}</p>}
          <div className="row">
            <button className="primary" type="submit" disabled={busy || !form.topic.trim()}>{busy ? "Starting..." : "Start"}</button>
            <span className="small muted">Mock media only (free placeholders). A budget counts LIST price: the Gemini free tier bills $0.</span>
          </div>
        </form>

        <div>
          <section className="panel">
            <h2>Setup</h2>
            {!health ? <p className="muted small">checking the API, Docker and the database...</p> : (
              <table className="small">
                <tbody>
                  <tr><td>LLM</td><td>{health.provider}: {health.model}, critic {health.critic_model}</td></tr>
                  <tr><td>API key</td><td>{health.key_set ? "set" : <strong>not set</strong>}</td></tr>
                  <tr><td>media</td><td>mock placeholders (free)</td></tr>
                  <tr><td>billing</td><td>{health.free_tier ? "Gemini free tier: $0 billed (costs shown are list price)" : "paid tier: list price is billed"}</td></tr>
                  <tr><td>research</td><td>{health.research_ready ? `ready (${health.vector_store?.backend ?? "?"}${health.vector_store?.rows ? `, ${health.vector_store.rows} rows` : ""})` : "off: the vector store is down"}</td></tr>
                  <tr><td>queue</td><td>{health.queue.running ? <Link href={`/runs/${health.queue.running}`}>{health.queue.running}</Link> : "idle"}
                    {health.queue.queued.length > 0 && ` + ${health.queue.queued.length} waiting`}</td></tr>
                </tbody>
              </table>
            )}
          </section>
          <section className="panel">
            <h2>Recent runs</h2>
            {!recent ? <p className="muted small">loading...</p> : recent.length === 0 ? <p className="muted small">none yet</p> : (
              <table className="small">
                <tbody>
                  {recent.map((r) => (
                    <tr key={r.id}>
                      <td><Link href={`/runs/${r.id}`}>{r.title ?? r.topic ?? r.id}</Link><div className="muted">{when(r.created)}</div></td>
                      <td><StatusBadge status={r.status} /></td>
                      <td className="num">{usd(r.cost_usd)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <p className="small"><Link href="/runs">All runs</Link></p>
          </section>
        </div>
      </div>
    </>
  );
}
