"""agents/writer.py: three options in one call, validated one by one, a pick call, prompt versions."""

import json
from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletionMessage

from shortsmith import llm
from shortsmith.agents import writer
from shortsmith.models import HookIdea, ResearchNotes

WORDS13 = "one two three four five six seven eight nine ten eleven twelve thirteen"


def option(title: str, words: str = WORDS13, language: str = "en", hook: str = "Stop scrolling now.") -> dict:
    scenes = [{"idx": i, "narration": words, "visual_prompt": f"p{i}", "duration_s": 5, "motion": "zoom_in"}
              for i in range(1, 7)]
    return {"title": title, "hook": hook, "scenes": scenes, "cta": "c", "language": language}


NOTES = ResearchNotes(trending=["a boy's cafe got 266,504 views"],
                      hooks=[HookIdea(pattern="POV", hook="POV: you found Tashkent's coziest autumn cafe.", why="puts you there")],
                      angles=["the sound of the steam wand"], avoid=["luxury"], visual_style="warm amber light, 35mm")


@pytest.fixture
def answers(monkeypatch):
    """The fake API answers from a queue; every request is kept."""
    queue: list[dict] = []
    seen: list[dict] = []

    def create(**request):
        seen.append(request)
        return NS(model="gemini-3.5-flash-lite", usage=NS(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None),
                  choices=[NS(message=ChatCompletionMessage(role="assistant", content=json.dumps(queue.pop(0))))])

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "MIN_INTERVAL_S", 0)
    monkeypatch.setattr(llm, "_client", lambda: NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setattr(llm.ledger, "record", lambda entry, path: None)
    monkeypatch.setattr(writer, "_library", {"library: Open loop (en)": "This cafe has no menu. The reason is genius."})
    return NS(queue=queue, seen=seen)


def test_options_are_validated_one_by_one_and_the_pick_sees_code_facts(answers):
    answers.queue += [
        {"options": [option("A"), option("B", words="too short"), option("C", words=" ".join(["so'z"] * 10), language="uz"),
                     option("D")]},
        {"scores": [], "best": 2, "reason": "D has the better hook"},
    ]
    result = writer.write("coffee shop", "en", NOTES, "v2")

    assert [o.title for o in result.options] == ["A"]  # B broke the word rule, C the language; D was a 4th option
    assert any("option 2" in r and "words in total" in r for r in result.rejected)
    assert any("option 3" in r and "'uz'" in r for r in result.rejected)
    assert result.pick is None and result.chosen.title == "A"  # one survivor: no pick call
    assert len(answers.seen) == 1


def test_the_pick_reads_spoken_words_and_measured_facts_only(answers):
    copied = option("B", hook="POV: you found Tashkent's coziest autumn cafe.")
    answers.queue += [{"options": [option("A"), copied]}, {"scores": [], "best": 2, "reason": "r"}]
    result = writer.write("coffee shop", "en", NOTES, "v2")

    assert result.chosen.title == "B" and len(result.results) == 2
    pick_request = answers.seen[1]
    assert pick_request["response_format"]["json_schema"]["name"] == "WriterPick"
    user = pick_request["messages"][1]["content"]
    assert "Option 1 (words: 78, target 78-84: in range; hook copied: no)" in user
    assert "hook copied: notes: POV" in user  # code found the copy, the model doesn't have to
    assert "p1" not in user  # no visual prompts: the pick is about what is said


def test_a_pick_off_the_list_falls_back_to_the_first(answers):
    answers.queue += [{"options": [option("A"), option("B")]}, {"scores": [], "best": 7, "reason": "r"}]
    result = writer.write("coffee shop", "en", None, "v1")
    assert result.chosen.title == "A" and "option 7" in result.pick.reason


def test_all_options_broken_twice_raises(answers):
    answers.queue += [{"options": [option("A", words="x")]}, {"options": [option("B", words="x")]}]
    with pytest.raises(ValueError, match="every script option broke a rule, 2 times"):
        writer.write("coffee shop", "en", None, "v1")


def test_v1_is_the_phase_5_prompt_and_v2_targets_its_weaknesses():
    assert writer.PROMPTS["v1"].startswith(llm.WRITER_SYSTEM_PROMPT)
    v2 = writer.PROMPTS["v2"]
    assert "13-14 words per scene" in v2 and "thrown away" in v2  # word counts
    assert "never reuse an example hook" in v2  # copied hooks
    assert "Never describe what the" in v2 and "visual_prompt only" in v2  # narration vs picture
    assert all(prompt.endswith(writer._OPTIONS_TASK) for prompt in writer.PROMPTS.values())


def test_v2_notes_give_patterns_without_ready_made_hooks_and_split_say_from_show():
    v1, v2 = writer.notes_text(NOTES, "v1"), writer.notes_text(NOTES, "v2")
    assert NOTES.hooks[0].hook in v1 and NOTES.hooks[0].hook not in v2
    assert "- POV: puts you there" in v2
    assert "The narrator SAYS the substance" in v2 and "the sound of the steam wand" in v2
    assert v2.endswith("warm amber light, 35mm")


def test_an_unknown_prompt_version_is_refused():
    with pytest.raises(ValueError, match="the versions are: v1, v2"):
        writer.write("t", "en", None, "v3")


@pytest.mark.parametrize("hook, copied", [
    ("POV: you found Tashkent's coziest autumn cafe.", "notes: POV"),
    ("pov - you found tashkent's coziest autumn cafe", "notes: POV"),  # case and punctuation don't hide a copy
    ("This cafe has no menu. The reason is genius!", "library: Open loop (en)"),
    ("POV: you found Samarkand's quietest tea house.", None),  # the pattern, adapted: not a copy
])
def test_copied_hooks_are_found_by_string_comparison(monkeypatch, hook, copied):
    monkeypatch.setattr(writer, "_library", {"library: Open loop (en)": "This cafe has no menu. The reason is genius."})
    assert writer.copied_from(hook, writer.hook_sources(NOTES)) == copied


def test_the_real_hook_library_has_about_200_examples():
    writer._library = None
    examples = writer.library_examples()
    assert 150 <= len(examples) <= 300 and any(name.endswith("(uz)") for name in examples)
