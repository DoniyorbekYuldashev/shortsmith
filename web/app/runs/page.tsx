"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { StatusBadge } from "@/components/StatusBadge";
import { ApiError, api } from "@/lib/api";
import { usd, when } from "@/lib/format";
import type { Problem, RunSummary } from "@/lib/types";

export default function Runs() {
  const router = useRouter();
  const [rows, setRows] = useState<RunSummary[]>();
  const [problem, setProblem] = useState<Problem>();

  useEffect(() => {
    const load = () => api.runs(100).then(setRows).catch((e) => setProblem(e instanceof ApiError ? e.problem : { message: String(e), hint: "" }));
    load();
    const timer = setInterval(load, 5000); // statuses move while a run works
    return () => clearInterval(timer);
  }, []);

  return (
    <>
      <h1>Runs</h1>
      <p className="muted">Newest first. Cost is list price from the ledger (billed $0 on the Gemini free tier; "-" = made
        before runs were tagged). Open one to replay it.</p>
      {problem && <div className="callout bad"><strong>{problem.message}</strong><p className="small">Next: {problem.hint}</p></div>}
      <div className="panel">
        {!rows ? <p className="muted">loading...</p> : rows.length === 0 ? (
          <div className="empty">No runs yet. <Link href="/">Make one</Link>.</div>
        ) : (
          <table>
            <thead>
              <tr><th>run</th><th>title</th><th>lang</th><th>status</th><th className="num">score</th><th className="num">length</th><th className="num">cost</th></tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id} className="link" onClick={() => router.push(`/runs/${r.id}`)}>
                  <td><Link href={`/runs/${r.id}`} className="mono">{r.id}</Link><div className="small muted">{when(r.created)}</div></td>
                  <td>{r.title ?? <span className="muted">{r.topic ?? "-"}</span>}</td>
                  <td>{r.language ?? "-"}</td>
                  <td><StatusBadge status={r.status} /></td>
                  <td className="num">{r.score ?? "-"}</td>
                  <td className="num">{r.length_s ? `${r.length_s.toFixed(1)} s` : "-"}</td>
                  <td className="num">{usd(r.cost_usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
