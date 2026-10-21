"""Count tokens locally with tiktoken, before anything is sent (and paid for).

gpt-4o and gpt-4.1 models share the `o200k_base` encoding (a vocabulary of ~200k pieces).
"""

import sys
from functools import cache
from pathlib import Path

import tiktoken

from shortsmith.llm import DEMO_SYSTEM_PROMPT

ENCODING = "o200k_base"

# Chat formatting wraps every message in hidden role markers, and a few more tokens
# prime the reply. OpenAI's cookbook gives these values for gpt-4o-family models;
# the result is an estimate, so compare it with the API's prompt_tokens.
TOKENS_PER_MESSAGE = 3
TOKENS_REPLY_PRIMER = 3

CONTEXT_WINDOW = {"gpt-4o-mini": 128_000, "gpt-4.1-mini": 1_047_576}  # input + output together

SAMPLES = {
    "English": "A cozy coffee shop in Tashkent is waiting for you this autumn.",
    "Uzbek (Latin)": "Toshkentdagi shinam qahvaxona bu kuzda sizni kutmoqda.",
    "Uzbek (Cyrillic)": "Тошкентдаги шинам қаҳвахона бу кузда сизни кутмоқда.",
    "Russian": "Уютная кофейня в Ташкенте ждёт вас этой осенью.",
}
DEMO_PROMPT = "Write a one-sentence hook about autumn coffee in Tashkent"  # the Step 1.1 test prompt


@cache
def _encoding() -> tiktoken.Encoding:
    """The first call downloads the vocabulary file once; tiktoken caches it on disk."""
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def token_pieces(text: str) -> list[str]:
    """Each token decoded on its own, to see where the tokenizer cuts words.

    BPE works on bytes, so one token can be half a letter: those show up as '�'.
    """
    enc = _encoding()
    return [enc.decode([token]) for token in enc.encode(text)]


def count_chat_tokens(messages: list[dict[str, str]]) -> int:
    """Estimate of the `prompt_tokens` the API will report for these messages."""
    total = TOKENS_REPLY_PRIMER
    for message in messages:
        total += TOKENS_PER_MESSAGE + sum(count_tokens(value) for value in message.values())
    return total


def _language_demo() -> None:
    english = count_tokens(SAMPLES["English"])
    print(f"{'language':<18}{'words':>6}{'tokens':>8}{'tok/word':>10}{'vs EN':>8}")
    for language, text in SAMPLES.items():
        n, words = count_tokens(text), len(text.split())
        print(f"{language:<18}{words:>6}{n:>8}{n / words:>10.2f}{n / english:>7.1f}x")
    for language in SAMPLES:
        print(f"\n{language}: " + " | ".join(token_pieces(SAMPLES[language])))


def _predict_demo_call() -> None:
    messages = [
        {"role": "system", "content": DEMO_SYSTEM_PROMPT},
        {"role": "user", "content": DEMO_PROMPT},
    ]
    text_only = count_tokens(DEMO_SYSTEM_PROMPT) + count_tokens(DEMO_PROMPT)
    print(f"\nStep 1.1 call: {text_only} tokens of text -> predicted prompt_tokens = {count_chat_tokens(messages)}")
    print("(run llm.py with credits and compare with its 'prompt tokens' line)")


def _file_report(paths: list[str]) -> None:
    window = CONTEXT_WINDOW["gpt-4o-mini"]
    for path in map(Path, paths):
        text = path.read_text(encoding="utf-8")
        n = count_tokens(text)
        print(
            f"{path.parent.name}/{path.name}: {len(text):,} chars -> {n:,} tokens "
            f"({len(text) / n:.1f} chars/token, {n / window:.1%} of gpt-4o-mini's window)"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        _file_report(sys.argv[1:])
    else:
        _language_demo()
        _predict_demo_call()
