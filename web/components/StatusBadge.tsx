import type { RunStatus } from "@/lib/types";

export function StatusBadge({ status }: { status: RunStatus }) {
  return <span className={`badge ${status.kind}`} title={status.detail || undefined}>{status.label}</span>;
}

const PHASE_LABEL = {
  connecting: "connecting", queued: "queued", running: "running", paused: "waiting for you",
  over_budget: "over budget", failed: "stopped by an error", done: "done",
} as const;

export function PhaseBadge({ phase }: { phase: keyof typeof PHASE_LABEL }) {
  return <span className={`badge ${phase}`}>{PHASE_LABEL[phase]}</span>;
}
