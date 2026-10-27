"""Which provider and model each role uses, which keys are set, and the free-allowance settings.

`shortsmith doctor` prints it and the web API's /health returns it, from this one function, so the
two never disagree. Nothing here sends a request, and a key is reported as set or not, never any
part of it.

The roles, and the settings that choose them (highest first):
- text (researcher, writer, director, revise): SHORTSMITH_<ROLE>_MODEL, else LLM_PROVIDER's default_model
- critic: --critic-model / SHORTSMITH_CRITIC_MODEL, else LLM_PROVIDER's critic_model
- embeddings (the RAG index): always gemini-embedding-2, whatever LLM_PROVIDER says (owner's choice, 2026-10-26)
- images: MOCK_MEDIA=1 -> placeholders; with --real, IMAGE_PROVIDER (openai / fal / mock) and its model settings
"""

import os

from pydantic import BaseModel

from shortsmith import costs, llm

KEYS = {  # what each key is for
    "OPENAI_API_KEY": "OpenAI: text (LLM_PROVIDER=openai), images (IMAGE_PROVIDER=openai)",
    "GEMINI_API_KEY": "Gemini: text (LLM_PROVIDER=gemini), and the RAG embeddings (always)",
    "FAL_KEY": "fal.ai: hero clips, and images with IMAGE_PROVIDER=fal (--real)",
    "YOUTUBE_API_KEY": "trend research (--research)",
}


class Role(BaseModel):
    role: str  # "text", "critic", "embeddings", "images (--real)"
    provider: str  # "openai", "gemini", "fal", "mock"
    model: str
    detail: str = ""  # e.g. the roles a text model serves, the vector size, the image quality and price
    key_env: str | None = None
    key_set: bool = True  # the key this role needs is set (True when it needs none)


class Setup(BaseModel):
    llm_provider: str
    roles: list[Role]
    keys: dict[str, bool]  # every key: set or not
    mock_media: bool  # MOCK_MEDIA: placeholders for every run unless --real
    allowance: costs.Allowance
    billing: str  # what "billed" means with these settings


def _key(provider: str) -> str:
    return llm.PROVIDERS[provider].key_env


def setup() -> Setup:
    """Raises ValueError for a setting with an unknown value (LLM_PROVIDER=nope): doctor and /health say so."""
    from shortsmith.agents import critic
    from shortsmith.media import images

    keys = {name: bool(os.environ.get(name, "").strip()) for name in KEYS}
    roles: list[Role] = []
    text: dict[str, list[str]] = {}
    for role in ("researcher", "writer", "director", "revise"):
        text.setdefault(llm.role_model(role), []).append(role)
    for model, served in text.items():
        p = llm.provider_of(model)
        effort = llm.reasoning_effort(model)
        roles.append(Role(role="text", provider=p, model=model, key_env=_key(p), key_set=keys[_key(p)],
                          detail=", ".join(served) + (f"; reasoning effort {effort}" if effort else "")))
    model = critic.critic_model()
    p = llm.provider_of(model)
    roles.append(Role(role="critic", provider=p, model=model, key_env=_key(p), key_set=keys[_key(p)],
                      detail="vision: judges 4 frames" + (f"; reasoning effort {e}" if (e := llm.reasoning_effort(model)) else "")))
    model = llm.embedding_model()
    p = llm.provider_of(model)
    roles.append(Role(role="embeddings", provider=p, model=model, key_env=_key(p), key_set=keys[_key(p)],
                      detail=f"{llm.EMBED_DIMENSIONS} numbers a vector; the RAG index, Gemini whatever LLM_PROVIDER says"))
    real = images.source(real=True)
    roles.append(Role(role="images (--real)", provider=real.provider, model=real.model, key_env=real.key_env,
                      key_set=keys[real.key_env] if real.key_env else True,
                      detail=(f"{real.quality}, {real.size}, ~${real.price_usd:.4f} an image" if real.quality else
                              f"~${real.price_usd:.4f} an image" if real.provider != "mock" else "placeholders even with --real")
                      + ("; now: placeholders (MOCK_MEDIA=1)" if images.mock_media() else "")))
    return Setup(llm_provider=llm.provider_name(), roles=roles, keys=keys, mock_media=images.mock_media(),
                 allowance=costs.allowance(), billing=costs.billing_note())
