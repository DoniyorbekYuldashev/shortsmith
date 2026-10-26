"""agents/researcher.py and the research notes: trends + retrieved hooks -> one LLM call -> notes.
YouTube, retrieval and the LLM are all faked."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletionMessage

from shortsmith import llm, make
from shortsmith.agents import researcher
from shortsmith.ledger import LedgerEntry
from shortsmith.models import NOTES_MAX, HookIdea, ResearchNotes
from shortsmith.research import rag, youtube

NOW = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)

NOTES_JSON = {
    "trending": ["A boy's homemade cafe got 266,504 views", "Bakery tours get 5-30k views"],
    "hooks": [{"pattern": "POV", "hook": "POV: you found Tashkent's quietest cafe", "why": "puts you inside"}],
    "angles": ["steam on a cold morning"],
    "avoid": ["the cheapest"],
    "visual_style": "",
}


def report(query: str, n: int, from_cache: bool = False) -> youtube.TrendReport:
    videos = [youtube.TrendVideo(video_id=f"v{i}", title=f"{query} video {i}", channel="c", views=1000 * (n - i),
                                 likes=None, comments=None, published=NOW, duration_s=30, vertical=True)
              for i in range(n)]
    return youtube.TrendReport(query=query, language="uz", published_after=NOW, fetched_at=NOW, candidates=n,
                               shorts_length=n, videos=videos, from_cache=from_cache, units_used=0 if from_cache else 101)


def hit(id: str, title: str, kind: str = "hook", group: str = "Curiosity hooks", similarity: float = 0.66) -> rag.Retrieved:
    return rag.Retrieved(id=id, kind=kind, title=title, group=group, text=f"Template: {title} ...", similarity=similarity)


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """YouTube, the knowledge base and the LLM, recorded."""
    seen: dict = {"searches": [], "retrieves": [], "requests": []}

    def search(query, language, max_results):
        seen["searches"].append((query, language, max_results))
        return report(query, 0 if query == "Samarqand Registoni kechqurun" else 8)

    def retrieve(query, k, kind="hook", brand=None):
        seen["retrieves"].append((query, k, kind, brand))
        if kind == "brand":
            return [hit(f"brands/{brand}/words", "Teahouse: Words we use and avoid", "brand", "Teahouse")]
        return [hit("story/pov", "POV"), hit("place/hidden-gem", "The hidden gem", similarity=0.65)]

    def create(**kwargs):
        seen["requests"].append(kwargs)
        return NS(model="gemini-3.5-flash-lite", usage=NS(prompt_tokens=1800, completion_tokens=300, prompt_tokens_details=None),
                  choices=[NS(message=ChatCompletionMessage(role="assistant", content=json.dumps(NOTES_JSON)))])

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(researcher.youtube, "search_trends", search)
    monkeypatch.setattr(researcher.rag, "retrieve", retrieve)
    monkeypatch.setattr(llm, "_client", lambda *_: NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setattr(llm.ledger, "record", lambda entry, path: seen.setdefault("ledger", []).append(entry))
    return seen


# --- the trend query ------------------------------------------------------------------------------


@pytest.mark.parametrize("topic, query", [
    ("coffee shop in Tashkent, cozy autumn vibe", "coffee shop in Tashkent"),
    ("Samarqand Registoni kechqurun", "Samarqand Registoni kechqurun"),
    ("street food: the best samsa", "street food"),
    ("plov — secrets from grandma", "plov"),
])
def test_the_trend_query_is_the_topics_first_part(topic, query):
    assert researcher.trend_query(topic) == query


def test_a_broader_query_drops_the_last_word_down_to_two():
    assert researcher.broader("Samarqand Registoni kechqurun") == "Samarqand Registoni"
    assert researcher.broader("Samarqand Registoni") is None


def test_too_few_shorts_means_one_broader_search(fakes):
    queries, found, error = researcher.find_trends("Samarqand Registoni kechqurun", "uz")
    assert queries == ["Samarqand Registoni kechqurun", "Samarqand Registoni"]
    assert found.query == "Samarqand Registoni" and len(found.videos) == 8 and error is None


def test_enough_shorts_means_one_search(fakes):
    queries, _, _ = researcher.find_trends("coffee shop in Tashkent, cozy", "en")
    assert queries == ["coffee shop in Tashkent"] and len(fakes["searches"]) == 1


def test_a_youtube_failure_is_reported_not_raised(monkeypatch):
    def broken(query, language, max_results):
        raise youtube.QuotaError("search: HTTP 403 quotaExceeded: ...")

    monkeypatch.setattr(researcher.youtube, "search_trends", broken)
    queries, found, error = researcher.find_trends("coffee shop", "en")
    assert found is None and "quotaExceeded" in error


# --- research() ---------------------------------------------------------------------------------


def test_research_is_one_structured_llm_call_over_the_gathered_data(fakes):
    r = researcher.research("coffee shop in Tashkent, cozy autumn vibe", "en", brand="teahouse")

    assert r.notes == ResearchNotes.model_validate(NOTES_JSON)
    [request] = fakes["requests"]
    assert request["response_format"]["json_schema"]["name"] == "ResearchNotes"
    system, user = request["messages"]
    assert system["content"] == researcher.RESEARCHER_SYSTEM_PROMPT
    text = user["content"]
    assert "Topic: coffee shop in Tashkent, cozy autumn vibe" in text
    assert "- 8,000 views | 30 s | coffee shop in Tashkent video 0" in text  # the evidence, compact
    assert "### POV\nTemplate: POV ..." in text and "### The hidden gem" in text
    assert "Brand rules (Teahouse):\n### Teahouse: Words we use and avoid" in text
    assert "https://" not in text and "0.66" not in text  # no links or scores: they cost tokens and help nobody
    brand_call, hook_call = fakes["retrieves"]  # the brand first: an unknown brand fails before anything is embedded
    assert brand_call == ("coffee shop in Tashkent, cozy autumn vibe: tone, words to use and avoid, visual style",
                          researcher.BRAND_K, "brand", "teahouse")
    assert hook_call == ("coffee shop in Tashkent, cozy autumn vibe", researcher.HOOKS_K, "hook", None)
    assert [e.step for e in fakes["ledger"]] == ["researcher"]
    assert (r.prompt_tokens, r.completion_tokens, r.trend_queries) == (1800, 300, ["coffee shop in Tashkent"])


def test_without_a_brand_no_brand_rules_are_retrieved(fakes):
    r = researcher.research("plov at home", "uz")
    assert r.brand_rules == [] and len(fakes["retrieves"]) == 1
    assert "Brand rules" not in fakes["requests"][0]["messages"][1]["content"]


def test_research_without_trends_says_so_and_goes_on(fakes, monkeypatch):
    monkeypatch.setattr(researcher.youtube, "search_trends",
                        lambda q, l, n: (_ for _ in ()).throw(youtube.YouTubeError("YOUTUBE_API_KEY is empty")))
    r = researcher.research("plov", "uz")
    assert r.trends is None and r.trend_error == "YOUTUBE_API_KEY is empty"
    assert "YouTube trends: none found (searches: 'plov')." in fakes["requests"][0]["messages"][1]["content"]


def test_the_summary_names_the_evidence_and_the_cost(fakes):
    text = researcher.summary(researcher.research("coffee shop", "en"))
    assert "8 videos for 'coffee shop' (live, 101 units)" in text
    assert "retrieved: POV 0.66, The hidden gem 0.65" in text and "1800 in + 300 out tokens" in text


# --- the notes ----------------------------------------------------------------------------------


def test_extra_items_are_cut_not_rejected():
    many = ResearchNotes(trending=[f"t{i}" for i in range(9)], angles=[f"a{i}" for i in range(9)],
                         avoid=[f"x{i}" for i in range(9)], visual_style="",
                         hooks=[HookIdea(pattern=f"p{i}", hook="h", why="w") for i in range(9)])
    assert (len(many.trending), len(many.hooks), len(many.angles), len(many.avoid)) == tuple(
        NOTES_MAX[k] for k in ("trending", "hooks", "angles", "avoid"))


def test_the_writer_reads_short_lines_and_the_style_only_when_there_is_one():
    notes = ResearchNotes.model_validate(NOTES_JSON)
    text = notes.for_writer()
    assert text.splitlines()[0].startswith("Research notes")
    assert '- POV: "POV: you found Tashkent\'s quietest cafe" (puts you inside)' in text
    assert "visual_prompt" not in text  # no brand: no style line
    styled = notes.model_copy(update={"visual_style": "warm amber light"})
    assert styled.for_writer().endswith("End every visual_prompt with these style words: warm amber light")
    assert llm.estimate_tokens(text) < 150


# --- make --research ----------------------------------------------------------------------------


def test_make_with_a_brand_researches_first_and_saves_the_research(fakes, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    written = {}

    def fake_write(topic, language, research=None):
        written["notes"] = research
        raise SystemExit("stop after writing")  # the media steps are tested elsewhere

    monkeypatch.setattr(make, "write_script", fake_write)
    monkeypatch.setattr("sys.argv", ["make", "coffee shop", "--brand", "teahouse"])
    with pytest.raises(SystemExit, match="stop after writing"):
        make.main()
    assert written["notes"] == ResearchNotes.model_validate(NOTES_JSON)
    assert fakes["retrieves"][0][3] == "teahouse"


def test_new_run_saves_research_json_next_to_the_script(fakes, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    r = researcher.research("coffee shop", "en")
    scenes = [{"idx": i, "narration": "w " * 13, "visual_prompt": "p", "duration_s": 5, "motion": "zoom_in"} for i in range(1, 7)]
    script = make.Script.model_validate({"title": "t", "hook": "h", "scenes": scenes, "cta": "c", "language": "en"})
    monkeypatch.setattr(make, "write_script", lambda topic, language, research=None: (script, None))
    run_dir = make.new_run("coffee shop", "en", r)
    saved = researcher.Research.model_validate_json((run_dir / "research.json").read_text(encoding="utf-8"))
    assert saved.notes == r.notes and saved.trends.query == "coffee shop"


def test_research_flags_are_refused_with_run(monkeypatch):
    monkeypatch.setattr("sys.argv", ["make", "--run", "runs/x", "--research"])
    with pytest.raises(SystemExit):
        make.main()


def test_the_ledger_entry_type_is_unchanged_for_old_lines():
    old = '{"ts":"2026-10-21T10:00:00Z","step":"writer","model":"gemini-3.5-flash-lite","prompt_tokens":1,"completion_tokens":1,"cost_usd":0.0}'
    assert LedgerEntry.model_validate_json(old).tokens_estimated is False  # lines written before Phase 5 still read
