"""How results look (Step 7.3): the tables and summaries cli.py prints on stdout.

cli.py decides what to do and when; this file only turns data (RunState, runs.RunDetail,
costs.CostReport, budget.RunEstimate) into lines and tables for a Ui (console.py), so the same
data can go to `--json`, or to Phase 8's web page, without touching the commands.
"""

import subprocess
from pathlib import Path
from typing import Any

from rich.text import Text

from shortsmith.console import STATUS_STYLE, Ui, usd


def estimate_lines(est: Any) -> list[str]:
    """The --real question: what would be paid for, in list price."""
    from shortsmith import costs

    source = f" ({est.image_source})" if est.image_source else ""
    lines = [f"images: {est.images} x {usd(est.images_usd / est.images if est.images else 0)} = {usd(est.images_usd)}{source}"]
    if est.clips:
        lines.append(f"hero clip: ~{usd(est.clip_usd, 2)} (asked again just before it is made)")
    if est.revision_images:
        lines.append(f"critic revisions: up to {est.revision_images} more images = {usd(est.revision_usd)} "
                     f"(+ a new clip if the hero picture changes, asked again)")
    lines.append(f"LLM: ~{usd(est.llm_usd)} list price ({costs.billing_note()})")
    lines.append(f"total: ~{usd(est.first_pass_usd)} for the first pass, ~{usd(est.most_usd)} at most")
    return lines


def kv(ui: Ui, key: str, value: str | Text, style: str | None = None) -> None:
    """One result line: a label column, then the value."""
    ui.print(Text.assemble((f"{key:<9}", "bold" if ui.rich_out else ""), value if isinstance(value, Text) else Text(value, style=style or "")))


def header(ui: Ui, s: Any, resumed: str | None = None) -> None:
    """make / resume's first lines (stderr): what this run is, its media and its budget."""
    from shortsmith import llm
    from shortsmith.agents import critic

    first = (f"resuming {s.run_id} before {resumed}" if resumed else f"run {s.run_id}") + f" | {s.topic!r} ({s.language})"
    extras = [f"prompt {s.prompt_version}", f"up to {s.max_rounds} revisions, pass at {s.pass_score:g}",
              f"hero clips {s.hero_clips}"]
    if s.research_on:
        extras.append(f"research{f' + brand {s.brand}' if s.brand else ''}")
    if s.via_mcp:
        extras.append("tools over MCP")
    ui.say(Text(f"{first} | {' | '.join(extras)}", style="bold"))
    from shortsmith.media import images

    media = f"REAL images, {images.source(real=True).label} (paid)" if s.media == "real" else "mock placeholders (free)"
    ui.say(f"media: {media} | LLM: {llm.model_for('writer')}, critic {critic.critic_model()}")
    if s.budget_usd is not None:
        ui.say(f"budget: {usd(s.budget_usd)} for the whole run, in LIST price (free tiers and free daily tokens bill $0; "
               "the budget counts what it would cost without them)")


def node_table(ui: Ui, s: Any, lines: list, estimates: dict[str, float]) -> None:
    """Per node: the ledger's calls, tokens and list price next to the estimate made before it ran."""
    from shortsmith import runs

    table = ui.table("node", "calls", "tokens in", "tokens out", "list price", "estimate", "seconds",
                     title=f"run {s.run_id}", right=("calls", "tokens in", "tokens out", "list price", "estimate", "seconds"))
    for row in runs.node_costs(lines, s):
        table.add_row(row.node, str(row.calls) if row.calls else "", f"{row.prompt_tokens:,}" if row.calls else "",
                      f"{row.completion_tokens:,}" if row.calls else "", usd(row.cost_usd) if row.calls else "",
                      usd(estimates[row.node]) if row.node in estimates and estimates[row.node] else "",
                      f"{row.seconds:.1f}" if row.seconds is not None else "")
    ui.print(table)


def money(ui: Ui, s: Any, lines: list, new: int, history: list | None = None) -> None:
    """The run's total, list price next to billed, and how much of its budget it used. `history`: the whole
    ledger (an OpenAI call's billing depends on that day's earlier calls)."""
    from shortsmith import costs

    list_usd, billed = sum(e.cost_usd for e in lines), costs.Billing(history or lines).total(lines)
    tokens = sum(e.prompt_tokens + e.completion_tokens + e.reasoning_tokens for e in lines)
    note = " (free tier / free daily tokens)" if billed == 0 and list_usd > 0 else ""
    kv(ui, "cost", f"{usd(list_usd)} list price | {usd(billed)} billed{note} | {len(lines)} calls ({new} in this "
                   f"session) | {tokens:,} tokens")
    if s.budget_usd is not None:
        kv(ui, "budget", f"{usd(list_usd)} of {usd(s.budget_usd)} used (list price)")


def finished(ui: Ui, s: Any, seconds: float) -> None:
    """Where the video is, how the critic scored it, and what to try next."""
    from shortsmith.agents import graph
    from shortsmith.media import ffmpeg

    final = Path(s.final_path).resolve() if s.final_path else None
    if final and final.exists():
        try:
            info = ffmpeg.probe(final)
            video = f"{final}  ({info.duration_s:.1f} s, {info.width}x{info.height}, {info.size_bytes / 1e6:.1f} MB)"
        except (OSError, ValueError, subprocess.CalledProcessError):
            video = str(final)
        kv(ui, "video", Text(video, style="bold green"))
    scores = " -> ".join(f"{r.score:g}" for r in s.critic_rounds)
    kv(ui, "critic", f"{scores or 'no review'} | {graph.stop_reason(s)}")
    kv(ui, "done", f"in {seconds:.0f} s. Details: shortsmith show {s.run_id} | redo a scene: "
                   f"shortsmith rerun-scene {s.run_id} <scene> --prompt \"...\"")


def status_text(ui: Ui, status: Any) -> Text:
    """A run's status, coloured in a terminal."""
    return Text(status.label, style=STATUS_STYLE.get(status.kind, "") if ui.rich_out else "")


def show_run(ui: Ui, d: Any) -> None:
    """`shortsmith show`: a runs.RunDetail as sections and tables."""
    ui.print(Text(f"{d.title or d.id} ({d.language or '?'})", style="bold"))
    kv(ui, "run", d.id + (f" | topic {d.topic!r}" if d.topic else ""))
    kv(ui, "status", status_text(ui, d.status) + Text(f" | {d.status.detail}" if d.status.detail else ""))
    if d.settings:
        st = d.settings
        kv(ui, "settings", f"prompt {st.prompt_version} | research {'on' if st.research_on else 'off'}"
                           f"{f' (brand {st.brand})' if st.brand else ''} | up to {st.max_rounds} revisions, pass at "
                           f"{st.pass_score:g} | hero clips {st.hero_clips} | media {st.media} | budget "
                           f"{usd(st.budget_usd) if st.budget_usd is not None else 'none'}")
    if d.hook:
        kv(ui, "hook", d.hook)
        kv(ui, "cta", d.cta or "")
    if d.hero:
        answer = {True: "approved", False: "refused (Ken Burns)", None: "not asked yet"}[d.hero.approved]
        kv(ui, "hero", f"scene {d.hero.scene}: {d.hero.motion_prompt} ({answer}; ~{usd(d.hero.cost_usd, 2)} with real media)")

    scenes = ui.table("#", "motion", "words", "voice", "narration", "picture prompt", title="Script", right=("#", "words", "voice"))
    for s in d.scenes:
        scenes.add_row(str(s.idx), s.motion, str(s.words), f"{s.voice_s:.1f} s" if s.voice_s else "-", s.narration, s.visual_prompt)
    ui.print(scenes)

    if d.critic_rounds:
        critic = ui.table("round", "score", "weak scenes (problem: reason -> fix)", "frames", "model", title="Critic", right=("round", "score"))
        for r in d.critic_rounds:
            weak = "\n".join(f"{i.scene} {i.problem}: {i.reason} -> {i.fix}" for i in r.weak_scenes) or "none"
            critic.add_row(str(r.round), f"{r.score:g}", weak, ", ".join(map(str, r.frames)) or "-", r.model)
        ui.print(critic)
    if d.revisions:
        revisions = ui.table("round", "scene", "problem", "before", "after", title="Revisions")
        for r in d.revisions:
            field = "narration" if r.narration_before != r.narration_after else "visual"
            before, after = (r.narration_before, r.narration_after) if field == "narration" else (r.visual_before, r.visual_after)
            revisions.add_row(str(r.round), str(r.scene), r.problem + ("" if r.changed else " (unchanged)"), before, after)
        ui.print(revisions)
    if d.reruns:
        reruns = ui.table("when (UTC)", "scene", "made", "kept in", "list price", title="Reruns", right=("scene", "list price"))
        for r in d.reruns:
            reruns.add_row(f"{r.at:%Y-%m-%d %H:%M}", str(r.scene), r.made, f"history/{r.label}", usd(r.cost_usd))
        ui.print(reruns)

    nodes = ui.table("node", "calls", "tokens in", "tokens out", "list price", "billed", "seconds", title="Cost per node",
                     right=("calls", "tokens in", "tokens out", "list price", "billed", "seconds"))
    for n in d.nodes:
        nodes.add_row(n.node, str(n.calls) if n.calls else "", f"{n.prompt_tokens:,}" if n.calls else "",
                      f"{n.completion_tokens:,}" if n.calls else "", usd(n.cost_usd) if n.calls else "",
                      usd(n.billed_usd) if n.calls else "", f"{n.seconds:.1f}" if n.seconds is not None else "")
    nodes.add_row("total", str(d.calls), "", f"{d.tokens:,} all", usd(d.cost_usd), usd(d.billed_usd), "")
    ui.print(nodes)

    f = d.files
    if f.final_mp4:
        info = f" ({f.video.duration_s:.1f} s, {f.video.width}x{f.video.height}, {f.video.size_bytes / 1e6:.1f} MB)" if f.video else ""
        kv(ui, "video", Text(f.final_mp4 + info, style="bold green"))
    else:
        kv(ui, "video", "no final.mp4 yet")
    kv(ui, "files", f"{f.images} images, {f.voices} voices, hero clips: {', '.join(f.hero_clips) or 'none'}, "
                    f"{f.critic_frames} critic frames | {', '.join(f.saved)}")
    if f.history:
        kv(ui, "history", ", ".join(f.history) + " (what each round or rerun replaced)")
    if d.traces:
        kv(ui, "traces", ", ".join(f"{t.trace_id} ({t.status}, {t.seconds:g} s)" if t.seconds is not None else t.trace_id
                                    for t in d.traces))
        kv(ui, "", "open: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000")


def cost_report(ui: Ui, report: Any, top: int) -> None:
    """`shortsmith cost`: a costs.CostReport as tables, then the quota view."""
    def table(title: str, keys: tuple[str, ...], rows: list) -> None:
        t = ui.table(*keys, "calls", "tokens in", "tokens out", "list price", "billed", title=title,
                     right=("calls", "tokens in", "tokens out", "list price", "billed"))
        for r in rows[:top]:
            first = (r.model, r.step) if len(keys) == 2 else (r.key,)  # separate cells: a "|" would break Markdown
            t.add_row(*first, str(r.calls), f"{r.prompt_tokens:,}", f"{r.completion_tokens:,}", usd(r.list_usd),
                      usd(r.billed_usd))
        if len(rows) > top:
            t.add_row(f"... {len(rows) - top} more (--top)", *[""] * (len(keys) + 4))
        ui.print(t)

    scope = f"since {report.since}" if report.since else "the whole ledger"
    ui.print(Text(f"Spending, {scope}{f', run {report.run}' if report.run else ''} (runs/ledger.jsonl)", style="bold"))
    table("By day (this computer's time zone)", ("day",), report.by_day)
    table("By run", ("run",), report.by_run)
    table("By model and step", ("model", "step"), report.by_model_step)
    t = report.total
    kv(ui, "total", f"{t.calls} calls | {t.prompt_tokens + t.completion_tokens:,} tokens | {usd(t.list_usd)} list price | "
                    f"{usd(t.billed_usd)} billed")
    kv(ui, "billed", report.billing)

    q = report.quotas
    quota = ui.table("model", "requests", "free daily limit", "left (by our count)",
                     title=f"Gemini today (since midnight Pacific = {q.day_start:%H:%M} UTC)",
                     right=("requests", "free daily limit", "left (by our count)"))
    for m in q.gemini:
        quota.add_row(m.model, str(m.requests), str(m.daily_limit) if m.daily_limit else "not measured",
                      str(max(0, m.daily_limit - m.requests)) if m.daily_limit else "-")
    ui.print(quota)
    kv(ui, "YouTube", f"{q.youtube_units:,} units today (our cap {q.youtube_cap:,}; Google's free quota {q.youtube_limit:,}; "
                      "a live search is 101 units, a cached one 0)")
    for note in q.notes:
        kv(ui, "note", note)

    o = q.openai
    free = ui.table("group", "models used today", "tokens", "free a day", "left (by our count)", "billed",
                    title=f"OpenAI free daily tokens (data sharing {'on' if o.data_sharing else 'OFF'}; since 00:00 UTC)",
                    right=("tokens", "free a day", "left (by our count)", "billed"))
    for g in o.groups:
        free.add_row(g.group, ", ".join(g.models) or "-", f"{g.tokens:,}", f"{g.daily_tokens:,}", f"{g.left:,}",
                     usd(g.billed_usd))
    ui.print(free)
    kv(ui, "past it", f"OPENAI_FREE_LIMIT_ACTION={o.action}: "
                      + ("a run stops before a node that would pass it" if o.action == "stop" else "a run warns and goes on"))
    for note in o.notes:
        kv(ui, "note", note)
