// The API client: one function per endpoint, typed with the generated types. The page calls the API
// directly (CORS allows http://127.0.0.1:3000 and http://localhost:3000), never through Next.js.
import type { Accepted, Health, Options, Problem, QueueView, RunDetail, RunSummary, StartRun } from "./types";

export const API = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8010").replace(/\/$/, "");

export class ApiError extends Error {
  readonly status: number;
  readonly problem: Problem;

  constructor(status: number, problem: Problem) {
    super(problem.message);
    this.status = status;
    this.problem = problem;
  }
}

type ValidationItem = { loc: (string | number)[]; msg: string };

// The API's refusals are {"detail": {"message", "hint"}}; FastAPI's own 422 is {"detail": [{loc, msg}]}.
function problemOf(status: number, body: unknown): Problem {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (detail && typeof detail === "object" && "message" in detail) return detail as Problem;
  if (Array.isArray(detail)) {
    const items = detail as ValidationItem[];
    return { message: items.map((d) => `${d.loc.at(-1)}: ${d.msg}`).join("; "), hint: "fix the form and try again" };
  }
  return { message: `the API answered ${status}`, hint: "see the terminal where `shortsmith serve` runs" };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API + path, {
      ...init,
      headers: init?.body ? { "Content-Type": "application/json" } : undefined,
      cache: "no-store",
    });
  } catch {
    throw new ApiError(0, {
      message: `can't reach the API at ${API}`,
      hint: "start it in the repo folder: uv run shortsmith serve",
    });
  }
  if (!res.ok) throw new ApiError(res.status, problemOf(res.status, await res.json().catch(() => null)));
  return (await res.json()) as T;
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });
const run = (id: string) => `/runs/${encodeURIComponent(id)}`;

export const api = {
  health: () => request<Health>("/health"),
  options: () => request<Options>("/options"),
  queue: () => request<QueueView>("/queue"),
  runs: (limit = 50) => request<RunSummary[]>(`/runs?limit=${limit}`),
  run: (id: string) => request<RunDetail>(run(id)),
  start: (body: StartRun) => request<Accepted>("/runs", post(body)),
  approve: (id: string, answer: "yes" | "no") => request<Accepted>(`${run(id)}/approval`, post({ answer })),
  resume: (id: string, budget_usd?: number) =>
    request<Accepted>(`${run(id)}/resume`, post(budget_usd === undefined ? {} : { budget_usd })),
};

export const eventsUrl = (id: string) => `${API}${run(id)}/events`;
export const fileUrl = (url: string) => API + url; // the events carry /runs/<id>/files/<path>?v=<round>
