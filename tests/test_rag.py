"""research/rag.py: chunking, the index and retrieval, with fake embeddings.

The fake embedding is a bag of words hashed into 64 numbers: texts that share words point in
similar directions, so retrieval behaves like the real thing, offline and deterministic.
The index tests run against every store (conftest.py's store_factory): an in-memory fake, Chroma
for real in a temporary folder, and pgvector in a throwaway table when the database is up.
"""

import hashlib
import math

import pytest

from shortsmith import llm
from shortsmith.research import rag, stores

DIMS = 64


def bag_of_words(text: str, scale: float = 1.0) -> list[float]:
    vector = [0.0] * DIMS
    for word in text.lower().replace("|", " ").replace(":", " ").split():
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS] += scale
    return vector


class FakeEmbed:
    """Stands in for llm.embed: counts the texts it was asked for."""

    def __init__(self, scale=lambda text: 1.0):
        self.calls: list[list[str]] = []
        self.scale = scale

    def __call__(self, texts: list[str]) -> llm.EmbedResult:
        self.calls.append(texts)
        return llm.EmbedResult(vectors=[bag_of_words(t, self.scale(t)) for t in texts], model="fake", dimensions=DIMS,
                               prompt_tokens=sum(len(t.split()) for t in texts), tokens_estimated=True,
                               cost_usd=0.0, latency_s=0.0)


HOOKS_MD = """# Curiosity hooks

An intro paragraph about the whole file: not a chunk.

## Open loop

**Template:** "[Strange fact]. Why? At the end."

**Examples:**
- en: "This cafe has no menu."

**When to use:** a strange concrete fact.

**Why it works:** an open question keeps people watching.

## Guess the price

**Template:** "Guess how much this costs."

**Examples:**
- en: "Guess the price of this melon."

**When to use:** markets and prices.

**Why it works:** everyone has a price intuition.
"""

PLACE_MD = """# Places

## Hidden gem

**Template:** "The best lagman in town has no sign."

**Why it works:** a hidden place feels like a gift.
"""

BRAND_MD = """# Teahouse Test (fictional)

## Voice and tone

Warm, calm, polite.

## Words we use and avoid

Use: cozy, guest. Avoid: cheapest, luxury.
"""


@pytest.fixture
def data(tmp_path):
    folder = tmp_path / "hooks"
    (folder / "brands").mkdir(parents=True)
    (folder / "curiosity.md").write_text(HOOKS_MD, encoding="utf-8")
    (folder / "place.md").write_text(PLACE_MD, encoding="utf-8")
    (folder / "brands" / "teahouse.md").write_text(BRAND_MD, encoding="utf-8")
    return folder


@pytest.fixture
def kb(data, store_factory):
    return rag.KnowledgeBase(data_dir=data, store=store_factory.open("kb"), model="fake", dimensions=DIMS, embed=FakeEmbed())


def stored(kb: rag.KnowledgeBase) -> dict[str, stores.Record]:
    return {r.id: r for r in kb.store.records()}


# --- chunking -----------------------------------------------------------------------------------


def test_split_sections_keeps_the_title_and_drops_the_intro():
    title, sections = rag.split_sections(HOOKS_MD)
    assert title == "Curiosity hooks"
    assert [heading for heading, _ in sections] == ["Open loop", "Guess the price"]
    assert sections[0][1].startswith('**Template:** "[Strange fact]')
    assert sections[1][1].endswith("everyone has a price intuition.")  # the last section is not lost
    assert "intro paragraph" not in "".join(body for _, body in sections)


def test_a_file_without_sections_has_no_chunks():
    assert rag.split_sections("# Only a title\n\nSome text.") == ("Only a title", [])


def test_chunks_get_stable_ids_kinds_and_their_brand(data):
    chunks = {c.id: c for c in rag.load_chunks(data)}
    assert set(chunks) == {"curiosity/open-loop", "curiosity/guess-the-price", "place/hidden-gem",
                           "brands/teahouse/voice-and-tone", "brands/teahouse/words-we-use-and-avoid"}
    hook, brand = chunks["curiosity/open-loop"], chunks["brands/teahouse/voice-and-tone"]
    assert (hook.kind, hook.title, hook.group, hook.brand) == ("hook", "Open loop", "Curiosity hooks", None)
    assert (brand.kind, brand.title, brand.brand) == ("brand", "Teahouse Test (fictional): Voice and tone", "teahouse")
    assert "**" not in hook.text and hook.text.startswith("Template:")  # markdown bold costs tokens, means nothing


def test_two_sections_with_the_same_id_are_an_error(data):
    (data / "place.md").write_text(PLACE_MD + "\n## Hidden  GEM!\n\nagain\n", encoding="utf-8")
    with pytest.raises(ValueError, match="place/hidden-gem"):
        rag.load_chunks(data)


def test_gemini_embedding_2_gets_its_task_formats_and_other_models_plain_text(data):
    chunk = rag.load_chunks(data)[0]
    assert rag.document_text(chunk, "gemini-embedding-2") == f"title: {chunk.title} | text: {chunk.text}"
    assert rag.query_text("cozy cafe", "gemini-embedding-2") == "task: search result | query: cozy cafe"
    assert rag.document_text(chunk, "text-embedding-3-small") == f"{chunk.title}\n{chunk.text}"
    assert rag.query_text("cozy cafe", "text-embedding-3-small") == "cozy cafe"


def test_each_model_and_size_gets_its_own_collection():
    assert rag.collection_name("gemini-embedding-2", 768) == "kb-gemini-embedding-2-768"
    assert rag.collection_name("text-embedding-3-small", 768) != rag.collection_name("text-embedding-3-small", 1536)


# --- the real knowledge base (data/hooks/) ---------------------------------------------------------


def test_the_knowledge_base_has_complete_hook_patterns_in_three_languages():
    chunks = rag.load_chunks(rag.DATA_DIR)
    hooks = [c for c in chunks if c.kind == "hook"]
    assert 50 <= len(hooks) <= 100
    for c in hooks:
        for label in ("Template:", "Examples:", "When to use:", "Why it works:", "- en:", "- uz:", "- ru:"):
            assert label in c.text, f"{c.id} has no {label!r}"
    brands = {c.brand for c in chunks if c.kind == "brand"}
    assert brands == {"anor-va-dolchin", "samoviy-karvon"}
    for brand in brands:
        sections = {c.title.split(": ", 1)[1] for c in chunks if c.brand == brand}
        assert {"Voice and tone", "Words we use and avoid", "Visual style"} <= sections


def test_every_chunk_fits_comfortably_in_one_embedding_request():
    sizes = rag.chunk_sizes(rag.load_chunks(rag.DATA_DIR))
    assert sizes["hook"]["max_tokens"] < 300 and sizes["brand"]["max_tokens"] < 300  # the model takes 8,192


# --- the index ------------------------------------------------------------------------------------


def test_the_first_sync_embeds_every_chunk_in_one_request(kb, data):
    report = kb.sync()
    assert (report.chunks, report.added, report.changed, report.removed, report.unchanged) == (5, 5, 0, 0, 0)
    assert len(kb.embed.calls) == 1 and len(kb.embed.calls[0]) == 5
    assert kb.store.count() == 5 and (report.store, report.collection) == (kb.store.backend, kb.store.name)
    record = stored(kb)["brands/teahouse/voice-and-tone"]  # everything retrieval returns is stored with the vector
    assert (record.kind, record.brand, record.group, record.source) == ("brand", "teahouse", "Teahouse Test (fictional)",
                                                                        "brands/teahouse.md")
    chunk = next(c for c in rag.load_chunks(data) if c.id == record.id)
    assert record.hash == rag.fingerprint(rag.document_text(chunk, "fake")) and len(record.embedding) == DIMS
    assert record.embedding == pytest.approx(bag_of_words(rag.document_text(chunk, "fake")))  # the vector, as embedded
    assert stored(kb)["curiosity/open-loop"].brand is None


def test_the_chroma_collection_records_its_model(data, tmp_path):
    store = stores.ChromaStore("kb-fake-64", DIMS, "fake", tmp_path / "chroma")
    rag.KnowledgeBase(data_dir=data, store=store, model="fake", dimensions=DIMS, embed=FakeEmbed()).sync()
    assert store.collection.metadata == {"embedding_model": "fake", "dimensions": DIMS} and store.name == "kb-fake-64"


def test_a_second_sync_with_no_edits_embeds_nothing(kb):
    kb.sync()
    report = kb.sync()
    assert (report.added, report.changed, report.unchanged, report.embedded_tokens) == (0, 0, 5, 0)
    assert len(kb.embed.calls) == 1  # still only the first sync's request
    assert kb.store.count() == 5  # no duplicates


def test_an_edited_section_is_re_embedded_and_a_deleted_one_removed(kb, data):
    kb.sync()
    edited = HOOKS_MD.replace("an open question keeps people watching", "curiosity is a craving")
    (data / "curiosity.md").write_text(edited, encoding="utf-8")
    (data / "place.md").unlink()
    report = kb.sync()
    assert (report.added, report.changed, report.removed, report.unchanged) == (0, 1, 1, 3)
    assert kb.embed.calls[-1] == [rag.document_text(next(c for c in rag.load_chunks(data) if c.id == "curiosity/open-loop"), "fake")]
    assert "place/hidden-gem" not in kb.store.hashes() and kb.store.count() == 4
    assert "curiosity is a craving" in stored(kb)["curiosity/open-loop"].text  # replaced in place, not added


def test_the_index_survives_a_restart(kb, data, store_factory):
    kb.sync()
    reopened = rag.KnowledgeBase(data_dir=data, store=store_factory.open("kb"), model="fake", dimensions=DIMS, embed=FakeEmbed())
    assert reopened.sync().unchanged == 5 and reopened.embed.calls == []


# --- retrieval --------------------------------------------------------------------------------------


def test_retrieve_returns_the_closest_hooks_first_with_their_similarity(kb):
    kb.sync()
    results = kb.retrieve("guess the price of this melon at the market", k=3)
    assert [r.id for r in results][0] == "curiosity/guess-the-price"
    assert all(r.kind == "hook" for r in results)  # brand sections are never mixed into hook results
    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)
    assert all(-1 <= s <= 1 for s in similarities)


def test_the_same_text_has_similarity_one(kb, data):
    kb.sync()
    chunk = next(c for c in rag.load_chunks(data) if c.id == "place/hidden-gem")
    [best] = kb.retrieve(rag.document_text(chunk, "fake"), k=1)
    assert best.id == "place/hidden-gem" and best.similarity == pytest.approx(1.0, abs=1e-4)


def test_cosine_ignores_how_long_the_vectors_are(data, store_factory):
    """Cosine compares directions. Doubling some vectors' length (as an unnormalised model might)
    changes no similarity; with L2 (straight-line) distance it would reorder the results."""
    plain = rag.KnowledgeBase(data_dir=data, store=store_factory.open("a"), model="fake", dimensions=DIMS, embed=FakeEmbed())
    stretched = rag.KnowledgeBase(data_dir=data, store=store_factory.open("b"), model="fake", dimensions=DIMS,
                                  embed=FakeEmbed(scale=lambda text: 3.0 if "melon" in text else 1.0))
    for kb in (plain, stretched):
        kb.sync()
    query = "guess the price of a melon in a hidden place"
    before, after = plain.retrieve(query, k=5), stretched.retrieve(query, k=5)
    assert [r.id for r in before] == [r.id for r in after]
    assert [r.similarity for r in before] == pytest.approx([r.similarity for r in after], abs=1e-3)


def test_brand_rules_come_only_from_that_brand(kb):
    kb.sync()
    results = kb.retrieve("which words to avoid", k=5, kind="brand", brand="teahouse")
    assert {r.id for r in results} == {"brands/teahouse/voice-and-tone", "brands/teahouse/words-we-use-and-avoid"}
    assert results[0].id == "brands/teahouse/words-we-use-and-avoid"


def test_an_unknown_brand_is_an_error_that_lists_the_brands(kb):
    kb.sync()
    with pytest.raises(ValueError, match="unknown brand 'nope'; the brands are: teahouse"):
        kb.retrieve("x", brand="nope", kind="brand")
    assert len(kb.embed.calls) == 1  # only the sync's: a bad brand costs no embedding request


def test_asking_for_more_than_there_is_returns_what_there_is(kb):
    kb.sync()
    assert len(kb.retrieve("anything", k=10)) == 3  # 3 hooks in the test data


def test_a_query_costs_one_embedding_request(kb):
    kb.sync()
    kb.retrieve("melon", k=2)
    assert kb.embed.calls[-1] == ["melon"]  # the fake model has no query format: plain text


def test_cosine_similarity_by_hand_matches_the_store(kb, data):
    """Every store's cosine distance is 1 - cos(angle): checked against the formula itself."""
    kb.sync()
    chunk = next(c for c in rag.load_chunks(data) if c.id == "curiosity/open-loop")
    a, b = bag_of_words("an open question about a strange fact"), bag_of_words(rag.document_text(chunk, "fake"))
    by_hand = sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))
    found = {r.id: r.similarity for r in kb.retrieve("an open question about a strange fact", k=3)}
    assert found["curiosity/open-loop"] == pytest.approx(by_hand, abs=1e-4)


def test_nearest_takes_a_vector_and_makes_no_request(kb):
    kb.sync()
    calls = len(kb.embed.calls)
    by_text = kb.retrieve("guess the price of this melon", k=2)
    by_vector = kb.nearest(bag_of_words("guess the price of this melon"), k=2)
    assert by_vector == by_text and len(kb.embed.calls) == calls + 1  # only retrieve() embedded


def test_no_filter_mixes_hooks_and_brand_sections(kb):
    kb.sync()
    kinds = {r.kind for r in kb.retrieve("cozy guest teahouse warm polite open loop", k=5, kind=None)}
    assert kinds == {"hook", "brand"}


def test_the_default_store_is_the_selected_one_and_never_a_fallback(data, monkeypatch, tmp_path):
    """VECTOR_STORE=pgvector (the default) without a database is an error with the next step;
    nothing quietly opens Chroma instead."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VECTOR_STORE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(stores.StoreUnavailable, match="DATABASE_URL is not set") as e:
        rag.KnowledgeBase(data_dir=data, model="fake", dimensions=DIMS, embed=FakeEmbed())
    assert "docker compose up -d db" in e.value.hint and "VECTOR_STORE=chroma" in e.value.hint
    assert not (tmp_path / "chroma").exists()  # no Chroma folder was made behind our back
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    kb = rag.KnowledgeBase(data_dir=data, model="fake", dimensions=DIMS, embed=FakeEmbed())
    assert kb.store.backend == "chroma" and kb.sync().added == 5 and (tmp_path / "chroma").exists()
