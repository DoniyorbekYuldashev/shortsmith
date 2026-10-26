// Numbers as the CLI prints them (console.usd): "$0" for nothing, else 4 decimals; list price throughout.
export function usd(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined) return "-";
  return value === 0 ? "$0" : `$${value.toFixed(digits)}`;
}

export const tokens = (n: number) => n.toLocaleString("en-US");

export function secs(value: number): string {
  if (value < 60) return `${value.toFixed(1)} s`;
  return `${Math.floor(value / 60)} min ${Math.round(value % 60)} s`;
}

export const NODE_LABEL: Record<string, string> = {
  researcher: "Researcher", writer: "Writer", director: "Director", approve_hero: "Approval", producer: "Producer",
  editor: "Editor", critic: "Critic", revise: "Revise",
};

// A run's `created` has no time zone (it's the folder name: this computer's time), and JavaScript reads a
// date-time without one as local time too, so the two agree.
export function when(iso: string | null | undefined): string {
  if (!iso) return "-";
  return new Date(iso).toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}
