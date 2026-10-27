# OpenAI switch build notes (side step after Phase 8: text and real images on OpenAI)

Written by the build session for the teaching session. Everything below happened on this machine (Windows 11
Home with Smart App Control on, Windows PowerShell 5.1, Python 3.12, uv 0.13.1, Node 24, Docker Desktop 29.4.0,
openai SDK as locked in `uv.lock`, no new packages) on 2026-10-26, with a scope change on 2026-10-27 (§5). Money spent: **$0**. OpenAI: one free
`models.list`, 2 chat requests refused with 429 (not billed, not in the ledger). Gemini: 0 calls. Images: 0.

**Done-check, in one paragraph.** The owner wants OpenAI for exactly two things: **text** on the data-sharing
free daily tokens, and **real images** from GPT Image 2. Both are built and tested offline. Every role has its
own model setting (text roles default to **gpt-5-mini**, the critic to **gpt-5.1** when `LLM_PROVIDER=openai`),
and a model always goes to its own provider's address and key. "Billed" understands **OpenAI's free daily
tokens** (`OPENAI_DATA_SHARING=1`: $0 within the day's allowance per model group, list price beyond, 00:00 UTC),
shown in `shortsmith cost`, `/health` and the web page, with an optional stop before a run passes it. Real
images have `IMAGE_PROVIDER` (default **openai gpt-image-2, low, 1024x1536 cropped to 9:16**; `fal` with Nano
Banana or FLUX; `mock`), only with `--real`; 0 were made. **The RAG embeddings stay on gemini-embedding-2 (768)
by the owner's choice**, whatever `LLM_PROVIDER` says: no embedding switch, no OpenAI index (§5). **Live check a
failed as the brief foresaw:** `429 insufficient_quota / credit_balance_exhausted` ("You have no credits
remaining"), so **the owner's $5 is not in the OpenAI org yet**; the mock run on OpenAI (b) and images (c) were
not run. Tests: **505 pass with the database** (474 before + 31), **483 pass + 22 skipped without it**; web:
typecheck, build, 9 tests. One optional exercise: `exercises/openai-switch/` (the allowance math, 5 tests).

**Three things the lesson should not miss:**
1. **"Free" is a running total, not a property of a call.** Whether an OpenAI call was billed depends on every
   call of its model group earlier that UTC day, so billing walks the WHOLE ledger in time order
   (`costs.Billing`), even when you only want one run's total. A per-line function (the old `billed_usd(entry)`)
   can't answer it (§3).
2. **A model name decides its provider.** `provider_of("gemini-...")` is Google, anything else OpenAI, so the
   writer can be on OpenAI while the critic or the RAG embeddings stay on Gemini, and a Gemini model name is
   never sent to OpenAI's address with OpenAI's key (§2, §5, D2).
3. **Prefix matching for prices was a silent-money bug waiting to happen.** With `gpt-5` in the price table,
   "longest prefix" would have priced `gpt-5.5` (4x dearer) and `gpt-5-pro` (12x) as gpt-5 without a word.
   Prices are now looked up by the exact model after cutting only a trailing snapshot date (§2, D7).

## 1. What was built (files)

| file | what changed |
|---|---|
| `llm.py` | `Provider.critic_model`; `provider_name()`, `provider_of(model)`, `ROLES`, `role_model()`, `model_for(step)`; `embedding_model()` (always Gemini's); `reasoning_effort(model)` (`OPENAI_REASONING_EFFORT`, default `low`); `_client(name)` one client per provider; `NoCreditError` (OpenAI's "no credit" 429, never retried); `min_interval()` (4.5 s spacing on Gemini only); a temperature for an OpenAI reasoning model fails before sending |
| `ledger.py` | OpenAI list prices (gpt-4.1/-nano, gpt-5/-mini/-nano, gpt-5.1, gpt-5.4/-mini/-nano, gpt-6-luna), `IMAGE_TOKEN_PRICES`, `IMAGE_PRICES`, Nano Banana in `MEDIA_PRICES`; `base_model()`; exact price lookup |
| `costs.py` | `FREE_GROUPS`, `FREE_DAILY_TOKENS`, `Allowance` / `allowance()`, `free_group()`, `allowance_day()` (UTC), `Billing` (the running total), `billing_note()`, `used_today()`, `allowance_today()` -> `GroupToday` / `AllowanceToday` in `Quotas.openai`; `CostReport.billing` |
| `budget.py` | `node_calls` per role; `same_calls` (exact model match for the history); images priced by `images.price_per_image()`; `RunEstimate.image_source`; `allowance_check()` + `FreeAllowanceReached` (warn or stop before a node) |
| `media/images.py` | `IMAGE_PROVIDER` (openai / fal / mock), `ImageSource` / `source()` / `price_per_image()`, `openai_price()`, the OpenAI path (`openai_request`, `usage_cost`, `fit_portrait`, `_openai`), `VerificationRequired` (the 403), fal by model (`fal_arguments`: Nano Banana 9:16 or FLUX's preset) |
| `providers.py` (new) | `setup()`: every role's provider and model, keys set or not, the allowance settings, the billing line: one source for `doctor` and `/health` |
| `agents/critic.py` | `CRITIC_MODEL` is now only the `--critic-model` override (None by default); `critic_model()` |
| `cli.py` | `explain()` cases for `VerificationRequired`, `NoCreditError`, `FreeAllowanceReached` (exit 4); `preflight()` checks each role model's own key, the image provider's key, and `FAL_KEY` when a hero clip may be made; `--real` names the image source; `doctor` rows per role + keys + billing + free tokens |
| `views.py` | estimate lines name the image source; billing note; the OpenAI allowance table in `cost` |
| `api.py`, `events.py`, `jobs.py` | `Health.billing`, `.setup`, `.allowance`; `RunStarted.billing`; the web still forces mock media |
| `runs.py` | billed figures from one whole-ledger `Billing` |
| `rag.py`, `migrate.py`, `mcp_server.py`, `graph.py`, `make.py`, `prompt_eval.py` | `llm.embedding_model()`, `critic.critic_model()`, the image source's label |
| `web/` | regenerated types; the home page's Setup table shows each role, the billing line and today's free tokens; Costs shows the billing line |
| `.env.example`, `README.md` | every new setting; a "Providers" table, how to switch for one run, the free-tokens paragraph |
| `tests/` | 32 new tests (§10); `conftest.py` clears provider settings a test could inherit from `.env` |
| `exercises/openai-switch/` | optional: `allowance_day` and `billed` (the running total) |

## 2. Defaults and why

| role | Gemini (`LLM_PROVIDER=gemini`) | OpenAI (`LLM_PROVIDER=openai`) | OpenAI list price per 1M (in / cached / out) |
|---|---|---|---|
| researcher, writer, writer-pick, director, revise | gemini-3.5-flash-lite | **gpt-5-mini**, reasoning effort low | $0.25 / $0.025 / $2.00 |
| critic (4 frames, vision) | gemini-3.8-flash | **gpt-5.1**, reasoning effort low | $1.25 / $0.125 / $10.00 |
| embeddings (the RAG index) | gemini-embedding-2 @768 | gemini-embedding-2 @768 (unchanged: §5) | free tier |
| images (`--real`) | n/a | **gpt-image-2, low, 1024x1536** (`IMAGE_PROVIDER`, default openai) | $5 text in, $30 image out; ~$0.005 an image |

Prices: developers.openai.com/api/docs/pricing, read on 2026-10-26 (the old `platform.openai.com/docs/pricing`
now redirects there).

**Why gpt-5-mini for the text roles.** (a) It is in the data-sharing **"mini" group** (2.5M free tokens a day at
usage tiers 1-2), listed there since August 2025 and still listed in 2026 popups users quoted. (b) Strict JSON
schemas and image input, both of which the pipeline needs. (c) No announced shutdown. The obvious alternative,
**gpt-4.1-mini** (fast, no reasoning, also "mini"), is announced for **shutdown on 2027-01-14** (OpenAI's notice of
2026-10-14, reported by several outlets): 2.5 months away, so not a default. **gpt-5.4-mini** is listed in the
mini group, but users on OpenAI's forum (March 2026) reported it **billed against the large group's cap**;
OpenAI support answered only that docs and coverage may be "misaligned". Not a default while that's unclear.
**gpt-6-luna** ($0.10 / $0.50, the recommended gpt-4.1-mini replacement) has no known group, so it isn't
known to be free; at its price a whole run's text would still cost under half a cent. It is priced in
`ledger.py`: `SHORTSMITH_WRITER_MODEL=gpt-6-luna` tries it.

**Why gpt-5.1 for the critic.** The Phase 5 rule (D14) is "stronger than the writer, and not the writer".
gpt-5.1 is a large model, a different one, sees images, and is named in the **"large" group** (250K a day at
tiers 1-2). A critic call is ~4.5k prompt tokens (about 3k of them the four frames) + ~1.5k out, so ~6k
tokens; a run makes at most 3: ~18k of the 250K. **Reasoning effort "low"** for every OpenAI reasoning model:
the default (medium) thinks longer and every thought is billed output; low is enough to follow a schema and a
rubric. It is a setting (`OPENAI_REASONING_EFFORT`), and is sent only to OpenAI's reasoning families
(gpt-5*, gpt-6*, o1/o3/o4), never to Gemini or a `*-chat-latest` alias.

**Day boundary: 00:00 UTC.** OpenAI's help article (help.openai.com/en/articles/10306912) answered 403 to our
fetch; third-party summaries of it agree the complimentary tokens refresh "daily at 00:00 UTC". That's what
`costs.allowance_day()` uses (Gemini's quota day stays midnight Pacific). In Tashkent (UTC+5) the OpenAI day
starts at 05:00 local time.

**Reasoning tokens stay right.** Gemini leaves its thinking out of `completion_tokens` and only adds it to
`total_tokens`, so `_complete` prices `total - prompt - completion` as output (Phase 6, E13). OpenAI's reasoning
models count their thinking **inside** `completion_tokens` (`completion_tokens_details.reasoning_tokens` says
how much) and `total = prompt + completion`, so the same rule gives 0 extra and nothing is counted twice. A test
pins it with a gpt-5-mini response of 1,000 in + 500 out (300 of them reasoning): priced 1,000 x $0.25 +
500 x $2.00 per 1M.

**Strict JSON schemas.** OpenAI's strict mode needs every object to list ALL its keys as `required` and set
`additionalProperties: false`, with no `allOf/not/if/then/else` (and the SDK's own strict converter also drops
`default`). All 7 schemas we send (ResearchNotes, Script, ScriptOptions, WriterPick, DirectorPlan,
CriticVerdict, SceneFixes) already comply: the Phase 1-6 models used `extra="forbid"` and no defaults in the
answer models. A test now checks every one against those rules, so a future field with a default fails offline
instead of at OpenAI with a 400.

## 3. The free-allowance accounting

Settings (all optional; defaults in brackets): `OPENAI_DATA_SHARING` [0], `OPENAI_FREE_DAILY_TOKENS_MINI`
[2,500,000], `OPENAI_FREE_DAILY_TOKENS_LARGE` [250,000], `OPENAI_FREE_LIMIT_ACTION` [warn | stop]. Data sharing
is **off by default**: until the owner has seen the Sharing page say "eligible", every OpenAI call shows as billed
at list price. The ledger has no OpenAI lines from before today (all 210 lines are Gemini), so turning the
setting on can't rewrite history.

**The groups** (`costs.FREE_GROUPS`, matched on the model without its snapshot date):
- mini: gpt-5.4-mini, gpt-5.4-nano, gpt-5.1-codex-mini, gpt-5-mini, gpt-5-nano, gpt-4.1-mini, gpt-4.1-nano,
  gpt-4o-mini, o4-mini, o3-mini, o1-mini, codex-mini-latest
- large: gpt-5.4, gpt-5.2, gpt-5.1, gpt-5.1-codex, gpt-5, gpt-5-codex, gpt-5-chat-latest, gpt-4.1, gpt-4o, o3, o1
- everything else is never free: embeddings, images, gpt-5.5, gpt-6-*, Gemini (which has its own free tier rule).

**The math** (`costs.Billing`): sort the whole ledger by time; per (UTC day, group) keep a running total of
input + output (+ Gemini-style reasoning) tokens; a call's free share is `max(0, min(tokens, daily - used_before))
/ tokens`; it is billed `list price x (1 - free share)`. Example from the tests, 1,000 free mini tokens:

| call | tokens | used before | free | billed |
|---|---:|---:|---:|---:|
| writer, $0.004 | 600 | 0 | 600 | $0 |
| director, $0.006 | 800 | 600 | 400 | $0.003 (half of its tokens are past the line) |
| revise, $0.002 | 200 | 1,400 | 0 | $0.002 |

The crossing call is split by tokens, not by price: OpenAI doesn't say how it bills a request that crosses
the line, and input and output tokens are priced differently, so this is an approximation for one call a day.

**Where it shows.** `shortsmith cost` (and `/costs`, `--json`) gets a table "OpenAI free daily tokens": per
group the models used today, tokens, the allowance, what's left by our count, and what today's calls beyond it
billed. Real output today with `OPENAI_DATA_SHARING=1` set for the command (no OpenAI calls yet):

```
OpenAI free daily tokens (data sharing on; since 00:00 UTC)
| group | models used today | tokens | free a day | left (by our count) | billed |
| mini  | -                 |      0 |  2,500,000 |           2,500,000 |     $0 |
| large | -                 |      0 |    250,000 |             250,000 |     $0 |
past it  OPENAI_FREE_LIMIT_ACTION=warn: a run warns and goes on
note     OpenAI's own counter can't be asked: these are the tokens OUR ledger recorded since 00:00 UTC (...).
         Calls made outside this project with the same organization (it is shared with Yukdosh) count there, not here.
note     Which models are free and how many tokens a day: only the organization's Data controls -> Sharing page
         says for sure (OPENAI_FREE_DAILY_TOKENS_MINI / _LARGE are settings, not facts).
```

`/health` returns the same (`allowance`), and the web's Setup table shows "free tokens: mini 0 of 2,500,000 ·
large 0 of 250,000 today (since 00:00 UTC, by our count)" when data sharing is on. Every "billed" figure (run
list, `show`, the run's cost line, the web's Done event) now comes from one `Billing` over the whole ledger.

**Before a node passes it** (`budget.allowance_check`, called by `budget.check` before every node): the node's
calls are estimated in tokens (the most of the last 10 such calls, else Phase 6's first-call guess) and added
to today's use per group. Past the allowance: `warn` logs a WARNING (the CLI shows it on stderr; the web server
logs it) and the run goes on; `stop` raises `FreeAllowanceReached` before anything is sent: the checkpoint of
the last finished node stays, the CLI prints one line with "wait for the next allowance (00:00 UTC), or let it
bill: OPENAI_FREE_LIMIT_ACTION=warn; then shortsmith resume RUN" and exits with **4** (like a budget stop); the
web shows it as a failed step with the same hint and a Resume button. `warn` is the default because the
amounts beyond the allowance are small (a whole run's text is ~$0.07 list) and our count can't see OpenAI's.

**The budget still counts list price** (`--budget`), as before: a whole mock run on OpenAI is ~$0.07 list
(the critic on gpt-5.1 is ~$0.02 a call), so `--budget 0.05` can stop a run that would be billed $0. That's the
budget's meaning since Phase 7 ("what it would cost without free tiers"); the run header now says so.

## 4. Real images from OpenAI (wired and tested offline; 0 made)

- `IMAGE_PROVIDER` picks the source **for real runs only**: `MOCK_MEDIA=1` (the default) and the web always make
  placeholders. `openai` is the default real source now; `fal` keeps working (`FAL_IMAGE_MODEL`:
  `fal-ai/nano-banana` by default, $0.0398, asked for `aspect_ratio: "9:16"`; `fal-ai/flux/schnell` keeps its
  `portrait_16_9` preset and ~$0.003); `mock` makes placeholders even with `--real`.
- The request (`images.openai_request`): `model=gpt-image-2, size=1024x1536, quality=low, n=1,
  output_format=png`, all settings (`OPENAI_IMAGE_MODEL`, `_SIZE`, `_QUALITY`). GPT Image models always answer
  with base64, so there is no download step.
- **The crop.** 1024x1536 is 2:3, the video is 9:16. `images.fit_portrait` centre-crops the picture to **864x1536**
  before saving: exactly what the editor's cover-and-crop would cut off the sides anyway, so the file on disk,
  the web's thumbnail, the critic's frame and the video all show the same picture. A 9:16 picture is left alone.
  (Custom sizes such as 864x1536 are documented for the gpt-image-2.5 models; for gpt-image-2 we kept the size
  every GPT Image model is documented to take.)
- **The price.** The response carries `usage` (input tokens, split into text and image, and output tokens), so
  each image is priced from what it really used: text in x $5 + image in x $8 + out x $30 per 1M
  (`ledger.IMAGE_TOKEN_PRICES`; every output token is priced at the image-output rate). Without usage, the
  per-image table is charged (`ledger.IMAGE_PRICES`, for 1024x1536: low $0.005, medium $0.041, high $0.165,
  OpenAI's reference figures as quoted by third parties; other sizes scaled by pixel count). The table is also
  what `--real`'s estimate and `--budget` use before anything is made. The ledger line is written **before** the
  picture is decoded or saved: it's paid for even if saving fails.
- **Verification.** A 403 that mentions verification becomes `images.VerificationRequired`, and the CLI says
  one line: `OpenAI refused gpt-image-2: the organization must be verified to use GPT Image models (403)` /
  `next: verify your organization in the OpenAI dashboard: Settings -> Organization -> General -> Verify
  Organization (access can take a few minutes after that); or IMAGE_PROVIDER=fal`. Nothing is recorded.
- **No dry check exists.** There's no API to ask "is this org verified?" without a generation request; an
  invalid request might answer 400 before checking verification, so it proves nothing. Left for the owner's go.
- `--real`'s question now names the source, e.g. `images: 6 x $0.0050 = $0.0300 (openai gpt-image-2 (low,
  1024x1536), ~$0.0050 an image)`. One real test video at low quality: ~$0.03 for 6 images, up to ~$0.06 if two
  critic rounds each redo 3 scenes. Medium: ~$0.25 for 6.
- `preflight` now checks the image provider's key, and `FAL_KEY` separately when a hero clip could be made (it
  is still made on fal): before, `FAL_KEY` covered both, and with OpenAI images a missing `FAL_KEY` would have
  failed only after the images were paid for.
- Hero clips: unchanged (mock, or fal after the approval question).

## 5. Embeddings stay on Gemini (scope change, 2026-10-27)

The first brief also asked for an `EMBEDDING_PROVIDER` switch, an OpenAI index and a Gemini-vs-OpenAI retrieval
comparison. **The owner dropped that part**: the RAG embeddings stay on gemini-embedding-2 at 768 numbers, in the
existing table `kb_gemini_embedding_2_768`. The first commit (33f9708) had built the switch and an
`eval_retrieval --models` mode (default Gemini; only the Gemini row was ever run, from stored vectors and cached
queries, 0 API calls; no OpenAI index was made). The follow-up commit removes them: `eval_retrieval.py` is back
to its pre-switch version, there is no `EMBEDDING_PROVIDER` or `SHORTSMITH_EMBEDDING_MODEL`, and the
text-embedding-3-large price is gone.

One thing had to stay, and it is the reason for a one-line change in `rag.py` and `migrate.py`: before this
step the index's model followed the chat provider (`llm.provider().embedding_model`), so `LLM_PROVIDER=openai`
would have silently built a new OpenAI index on the first `--research` run. Now `llm.embedding_model()` always
returns Gemini's model, and `provider_of()` sends it to Google's address with Google's key, so text on OpenAI
and the index on Gemini live side by side. Consequence: `--research` needs `GEMINI_API_KEY` even with
`LLM_PROVIDER=openai` (preflight and doctor check it). A test pins both halves
(`test_the_rag_embeddings_stay_on_gemini_whatever_the_text_provider`). `PROVIDERS["openai"].embedding_model` and
its price stay, untouched, with the rest of the OpenAI path.

## 6. Wiring: doctor, /health, the web

`shortsmith doctor` on this machine with `LLM_PROVIDER=openai` set for the command (real output, keys never
printed):

```
| LLM_PROVIDER    | -  | openai
| text            | ok | openai: gpt-5-mini (researcher, writer, director, revise; reasoning effort low)
| critic          | ok | openai: gpt-5.1 (vision: judges 4 frames; reasoning effort low)
| embeddings      | ok | gemini: gemini-embedding-2 (768 numbers a vector; the RAG index, Gemini whatever LLM_PROVIDER says)
| images (--real) | -  | openai: gpt-image-2 (low, 1024x1536, ~$0.0050 an image; now: placeholders (MOCK_MEDIA=1))
| OPENAI_API_KEY  | ok | set; OpenAI: text (LLM_PROVIDER=openai), images (IMAGE_PROVIDER=openai)
| GEMINI_API_KEY  | ok | set; Gemini: text (LLM_PROVIDER=gemini), and the RAG embeddings (always)
| FAL_KEY         | -  | set; fal.ai: hero clips, and images with IMAGE_PROVIDER=fal (--real)
| YOUTUBE_API_KEY | -  | set; trend research (--research)
```

A key row is `ok`/`FAIL` when a text, critic or embeddings model needs it, information otherwise; a role row
fails when its model has no list price. `/health` adds `billing`, `setup` (the same roles, keys as true/false,
the allowance settings and which of them are set) and `allowance` (today's use). The web page's Setup table shows
one row per role, the billing line and today's free tokens; the web stays mock-only whatever `MOCK_MEDIA` or
`IMAGE_PROVIDER` say (a test sets both to real values and checks the run is mock).

## 7. Live checks

**a. One tiny text call on OpenAI: FAILED, as the brief anticipated.** `LLM_PROVIDER=openai` was set as a
process variable for that one command (`.env` untouched; uv's `--env-file` did not override it: the request
went to OpenAI). OpenAI's raw answer:

```
Error code: 429 - {'error': {'message': 'You have no credits remaining. Add credits to continue using the API at
https://platform.openai.com/settings/organization/billing/.', 'type': 'insufficient_quota', 'param': None,
'code': 'credit_balance_exhausted'}}
```

Our side turned it into `NoCreditError: the OpenAI organization has no credit left (429 insufficient_quota)`,
the CLI's hint is "add credit in the OpenAI dashboard (Settings -> Billing), then ...; or LLM_PROVIDER=gemini",
`paced()` doesn't retry it, and 0 ledger lines were added. The call still took **3.8 s**: the OpenAI SDK's own
two retries (it retries every 429 unless the server says not to). **Stopped here, per the brief: b and c not run.**

The teaching session probed the same key afterwards (free, $0 balance): chat gpt-4.1-mini, gpt-image-2 and
gpt-image-1-mini all answered `429 credit_balance_exhausted / insufficient_quota`. The key and the org are valid,
and the balance check comes before any verification check, so **whether the org is verified for GPT Image is
unknown until the owner pays**; the first image request after the go will say (a 403 becomes the one-line
"verify your organization" error, §4).

**When the $5 is in** (terminal, repo root, PowerShell; refresh PATH first), the remaining checks are:

```powershell
# a. one tiny call (prints model, tokens, price, seconds)
$env:LLM_PROVIDER = "openai"; uv run --env-file .env python -m shortsmith.llm "coffee shop in Tashkent"; Remove-Item Env:LLM_PROVIDER
# b. one full mock-media run on OpenAI (~10 calls: researcher off, writer 2, director 1, critic <=3, revise <=2)
$env:LLM_PROVIDER = "openai"; uv run shortsmith make "coffee shop in Tashkent" --approve no; Remove-Item Env:LLM_PROVIDER
```

**Speed: Gemini's side of the table** (from runs already in the ledger and checkpoints, `shortsmith show`),
OpenAI's column to be filled by b:

| node | Gemini run 2026-10-25_144556 (s) | Gemini run 2026-10-24_182435 (s) | OpenAI (s) |
|---|---:|---:|---:|
| researcher | 8.4 | 8.7 | - |
| writer (options + pick) | 6.2 | 13.1 | - |
| director | 5.7 | 5.0 | - |
| producer (mock) | 8.0 | 22.1 | - |
| editor (ffmpeg) | 50.8 | 135.1 | - |
| critic (all rounds) | **499.4** (2 calls) | **567.2** (3 calls) | - |
| revise | 13.7 | 7.8 | - |

The critic is ~85% of a Gemini run's wall time (free-tier queueing on Flash; CLAUDE.md's 13-250 s per turn).
That's where OpenAI should win; the editor is ffmpeg and doesn't change. Expected cost of one OpenAI mock run at
list price: ~$0.07 (critic ~$0.02 a call on gpt-5.1, the rest under $0.01), ~7k mini + ~18k large tokens:
**billed $0** within the allowance, once data sharing is on and eligible.

### 7b. After the owner's credit (2026-10-27)

The owner added credit (balance about $5.99), the organization shows **Verified** (Settings -> Organization ->
Verifications), and the org's Sharing page says **"You're enrolled for complimentary daily tokens"**, enabled for
the shortsmith project. So `OPENAI_DATA_SHARING=1` is now right to set (the owner flips `.env`).

**Live check a: PASSED.** One call with `LLM_PROVIDER=openai` set for that command only (`.env` untouched):
`gpt-5-mini-2025-08-07` answered "OK", **13 tokens in + 10 out, $0.000023 list price, 3.09 s**, one ledger line
(step `openai-check`). Reasoning effort `low` was accepted.

**Live check b: PASSED** (owner's go, 2026-10-27). One mock-media run with `LLM_PROVIDER=openai`,
`OPENAI_DATA_SHARING=1` and `MOCK_MEDIA=1` set for that command only: `shortsmith make "coffee shop in Tashkent,
cozy autumn vibe" --approve no --no-trace`, run **2026-10-27_132453**, final video
`runs/2026-10-27_132453/final.mp4` (32.5 s, 1080x1920). **4 calls, 12,036 tokens, $0.0222 list price, $0 billed;
done in 124 s** (140 s wall, uv start included). The critic (gpt-5.1) gave **7.8** on the first look and the run
passed (7.8 >= 7): no revision round, so no revise call. It still named scenes 1, 4 and 6 as weak, from the
placeholder frames of scenes 1, 3, 4 and 6.

| node | OpenAI run 2026-10-27_132453 | Gemini run 2026-10-25_144556 | Gemini run 2026-10-24_182435 |
|---|---|---|---|
| writer (options + pick) | **29.4 s**, 2 calls, 1,495 in / 4,330 out, $0.0090 | 6.2 s | 13.1 s |
| director | **7.0 s**, 847 in / 926 out, $0.0021 | 5.7 s | 5.0 s |
| producer (mock) | 26.9 s | 8.0 s | 22.1 s |
| editor (ffmpeg) | 46.7 s | 50.8 s | 135.1 s |
| critic | **14.0 s** (1 call), 3,802 in / 636 out, $0.0111 | **499.4 s** (2 calls) | **567.2 s** (3 calls) |
| revise | not needed (passed) | 13.7 s | 7.8 s |
| whole run | **124 s** | about 10 min | about 13 min |

What it shows:
- **The critic is where the time went, and it's 35x faster per call**: 14 s for one gpt-5.1 call against ~250 s
  for one Gemini Flash call (499 s / 2) on the free tier. That alone turns a 10-13 minute run into about 2 minutes.
- **The writer is slower on OpenAI** (29 s against 6-13 s): gpt-5-mini writes 3 options and thinks first. Its
  3,910 completion tokens include reasoning (OpenAI counts it inside `completion_tokens`); flash-lite wrote less.
  If the writer's 29 s matters, `SHORTSMITH_WRITER_MODEL` can try a faster model; nothing else needs it.
- **Estimates held**: the critic was estimated at $0.0206 (the first-call guess: 4,500 in + 1,500 out) and cost
  $0.0111; the writer $0.0111 estimated, $0.0090 real.
- Producer and editor are local (placeholders, edge-tts, ffmpeg), so their times vary with the machine, not the provider.

**Today's free tokens after the run** (`shortsmith cost` with `OPENAI_DATA_SHARING=1`; our ledger since 00:00 UTC):

| group | models used today | tokens | free a day | left (by our count) | billed |
|---|---|---:|---:|---:|---:|
| mini | gpt-5-mini | 16,015 | 2,500,000 | 2,483,985 | $0 |
| large | gpt-5.1 | 4,438 | 250,000 | 245,562 | $0 |

Today's OpenAI total: **8 calls, 20,453 tokens, $0.0349 list price, $0 billed.** That is check a (1 call) + this
run (4) + **3 calls from run 2026-10-27_121145**: the first attempt at check b (07:12 UTC) had started before it
was stopped, and it made the writer and director calls (725 + 5,071, 778 + 313, 846 + 661 tokens; $0.0127 list,
$0 billed) before it ended. That run is unfinished: its folder and checkpoint are still there
(`shortsmith resume 2026-10-27_121145` would finish it; nothing else depends on it). Total text calls this
session: 8 of the ~25 allowed. Whether OpenAI's own counter agrees can only be seen on the org's Usage page.

**Images: 0 made.** An image go arrived and was cancelled again by the owner before anything was sent.

## 8. Errors (every one, how it was found, the fix)

**E1. The first two full test runs hung at the first web test.** Found: `pytest -v` stopped at
`test_api.py::test_a_run_streams_its_events_in_order_and_ends_with_done` for 10 minutes. Cause: 22 tests replace
`llm._client` with `lambda: fake`, and the new code calls `_client("gemini")`: TypeError inside the web worker's
thread, where the test then waited (I didn't trace why it waited instead of failing). Fix: the fakes take
`lambda *_:`; the test alone passed in 6 s, the suite in 54 s. Lesson: a signature change to a function tests
monkeypatch is a change to every fake.

**E2. 9 failures after the switch, all from new defaults, none a bug.** Found: the full offline run. FLUX-priced
expectations (`6 x $0.0030`) now meet gpt-image-2 (`6 x $0.0050`); `MOCK_MEDIA=0` tests needed `FAL_KEY` only and
now needed `OPENAI_API_KEY`; the paced test expected Gemini's 4.5 s spacing with `LLM_PROVIDER` unset (= OpenAI
now: no spacing); the CLI's cost line wording changed; the OpenAPI types needed `npm run types`. Fix: each test
now names the provider it assumes (fal + FLUX keeps the old numbers), plus new tests for the OpenAI defaults.

**E3. Two new tests failed only with `.env` loaded** (`uv run --env-file .env pytest`): `.env`'s
`YOUTUBE_API_KEY` made `/health`'s keys dict differ, and its `LLM_PROVIDER=gemini` gave the no-credit test
4.5 s spacing, which the test reads as "it retried". Fix: those tests set or delete what they assume, and
`conftest.py` now clears every new provider setting before each test (`no_provider_settings`).

**E4. Two new budget tests were wrong, not the code.** The billing line depends on `LLM_PROVIDER` (the fixture
sets gemini), and the critic's first-call guess is 4,500 + 1,500 = 6,000 tokens, more than the 5,000 "large"
allowance the test gave it. Fix: the test sets `LLM_PROVIDER=openai` and a 10,000 allowance.

**E5. The web typecheck failed on the test fixture**: `Property 'billing' is missing ... but required`. The
Python field has a default (`billing: str = ""`), yet FastAPI's response schema marks it required (a response
always contains it), so the generated TypeScript type does too. Fix: the fixture sets it. Worth knowing:
"has a default" means optional in a request schema and required in a response schema.

**E6. A Chroma test failed once** (`test_rag.py::test_cosine_ignores_how_long_the_vectors_are[chroma]`, `Error
creating hnsw segment reader: Nothing found on disk`): the known flaky one (Phase 8 notes, E15); it passed 3 of 3
alone and in every later full run. Not touched.

**E7. Live check a: 429 `credit_balance_exhausted`** (§7): expected, handled, the reason the live part stops here.

**E8. Scope change after the first commit (2026-10-27).** The owner dropped the OpenAI-embeddings part
(§5). Found: a message from the teaching session. Fix: the switch and the `--models` evaluation were removed in a
follow-up commit, the embeddings pinned to Gemini, one test replaced (506 -> 505 with the database).

**E9 (tooling, not the code).** A bash heredoc with a Python test containing `can't` in a string broke the shell's
quoting (`unexpected EOF while looking for matching '`); the new tests were written to a file and appended instead.

## 9. Design decisions

- **D1. Defaults: gpt-5-mini for text, gpt-5.1 for the critic** (§2): in the free groups, current, not the writer.
- **D2. The model name decides the provider** (`provider_of`): one rule, no per-role provider settings, mixing is
  possible (OpenAI writer + Gemini critic), and a model can't be sent to the wrong address.
- **D3. Per-role settings follow the existing name:** `SHORTSMITH_<ROLE>_MODEL`, like Phase 6's
  `SHORTSMITH_CRITIC_MODEL`; unset = the provider's default, so `LLM_PROVIDER` alone switches everything.
- **D4. `--critic-model` keeps working** through `critic.CRITIC_MODEL` (now only the override, None by default),
  read through `critic.critic_model()`. Its help text names both providers' defaults.
- **D5. Pacing only on Gemini** (`min_interval()`): OpenAI's limits don't need 4.5 s between calls.
- **D6. "No credit" is its own error and never retried**: waiting can't fix it; the hint says where to add credit.
- **D7. Exact price lookup** after cutting a trailing `-YYYY-MM-DD` (`ledger.base_model`), instead of the
  longest prefix; the budget's history matches models the same way (`budget.same_calls`).
- **D8. Billing is computed when reported, from the whole ledger** (not stored on the line): the ledger stays a
  record of facts (tokens, list price), and changing a setting (data sharing on, a different allowance) changes
  the view, not the history. Cost: `Billing` reads every line; at ~200 lines that's nothing.
- **D9. Data sharing off by default; the groups and amounts are settings or a documented table**, because only
  the org's Sharing page knows.
- **D10. Past the allowance: warn by default, stop on request**, stopping like a budget stop (exit 4, resumable).
- **D11. The RAG embeddings are always Gemini's** (owner's choice): `llm.embedding_model()` has no setting, and
  the index doesn't move when the chat provider does (§5). An embedding switch was built, then removed.
- **D12. Images: openai by default for real runs, mock unless `--real`, crop on save, price from usage**, ledger
  line before the picture is decoded.
- **D13. `providers.py` for doctor and /health**: one function, so the two can't disagree.
- **D14. A temperature for an OpenAI reasoning model fails before sending** (they refuse any temperature but 1).
  Only `experiments.py` (Phase 1's temperature test) sends one: run it with `LLM_PROVIDER=gemini`, or on a
  non-reasoning OpenAI model such as gpt-4.1-mini.

## 10. Tests

New (31): `test_llm.py` 10 (roles and defaults; a Gemini critic reaches Gemini's client while
`LLM_PROVIDER=openai`; reasoning effort only for OpenAI reasoning models; reasoning inside completion priced once;
no credit not retried and not recorded; spacing only on Gemini; all 7 schemas meet OpenAI's strict rules; the
RAG embeddings stay on Gemini and reach Gemini's client whatever LLM_PROVIDER says; exact prices for dated names and none for gpt-5.5 / gpt-5-pro / gpt-5.1-codex-max; temperature refused for a
reasoning model), `test_budget.py` 8 (free within / billed beyond, the crossing call's share; per group and
00:00 UTC, Tashkent's 04:59 still the previous day; off = list price; models outside the groups; the allowance view;
the report bills $0; warn / stop / off before a node; the whole-run estimate with gpt-image-2 low and medium),
`test_images.py` 7 (the source follows MOCK_MEDIA then IMAGE_PROVIDER, sizes scale the estimate; the request and
the 864x1536 crop; price from usage and the ledger line; the table when usage is missing; the 403 -> one line +
next step, nothing recorded or saved; `fit_portrait`; Nano Banana asks for 9:16), `test_cli.py` 4 (doctor rows
per role, both keys needed, never a key; an unknown IMAGE_PROVIDER; NoCreditError and FreeAllowanceReached
messages and exit 4; `--real` needs FAL_KEY for a hero clip and the image provider's key), `test_api.py` 2
(`/health` roles, keys as booleans, today's allowance, no secret anywhere; the web stays mock with
`MOCK_MEDIA=0` and `IMAGE_PROVIDER=openai`).

Results: `uv run pytest` **483 passed, 22 skipped**; `uv run --env-file .env pytest` **505 passed**; web
`npm run typecheck` clean, `npm run build` ok, `npm test` 9 passed; `uv run pytest exercises/openai-switch`
5 fail with NotImplementedError by design (a reference solution passes all 5).

## 11. Honest limits

- **We can't see OpenAI's free-token counter.** Our "left today" is our ledger's sum; OpenAI's may be lower:
  refused or failed requests, other tools, and **Yukdosh** (same org) all count there and not here. If Yukdosh
  restarts on this org it would use the same allowance and the same $5.
- **Eligibility and the model list are only on the org's Sharing page** (Data controls -> Sharing). The group
  table is from OpenAI's help page and users' quotes of that page (2025-2026); gpt-5.4-mini's group was disputed
  in March 2026. Check the page, then set `OPENAI_DATA_SHARING=1` and the two amounts (tiers 3-5 get 10M / 1M).
- **Data sharing means OpenAI may train on these prompts and answers.** Only topics, scripts, hook patterns and
  the brand guidelines in `data/hooks/` go in: no private or company data. Keep it that way (PLAN §3's note).
- **The crossing call's split is our approximation** (tokens, not price); OpenAI doesn't document it.
- **gpt-image-2's per-image table** comes from third-party quotes of OpenAI's calculator (one site measured
  higher bills); real images are priced from the response's usage, so the first real image settles it.
- **gpt-4.1-mini disappears on 2027-01-14**; nothing defaults to it, but Phase 1's comparison notes name it.
- **Nothing live on OpenAI ran**: strict schemas, vision frames and reasoning effort are checked against the
  documented rules and fakes, not yet against the API.

## 12. Suggested CLAUDE.md / PLAN.md changes (the teaching session edits those)

**CLAUDE.md, Gotchas:**
- After the $5: switch text with `LLM_PROVIDER=openai` (defaults gpt-5-mini; critic gpt-5.1; reasoning effort
  low). Turn on `OPENAI_DATA_SHARING=1` only after the Sharing page says eligible (and set the two
  `OPENAI_FREE_DAILY_TOKENS_*` to what it shows). OpenAI's free day starts 00:00 UTC (05:00 in Tashkent).
- The RAG embeddings are always gemini-embedding-2, whatever `LLM_PROVIDER` says: `--research` needs
  `GEMINI_API_KEY` even with text on OpenAI.
- Real images: `IMAGE_PROVIDER=openai` (default for `--real`), `OPENAI_IMAGE_QUALITY=low` (~$0.005) / medium
  (~$0.041); 1024x1536 is cropped to 864x1536. A 403 means "verify the organization". Nano Banana stays
  available with `IMAGE_PROVIDER=fal` (`FAL_IMAGE_MODEL`). `--real` needs `FAL_KEY` too unless `--hero-clips 0`.
- "Keep the OpenAI path intact" can become "the OpenAI path is the fast path"; the `credit_balance_exhausted`
  note stays true until the $5 lands (still true on 2026-10-26, checked by this build).
- `uv run --env-file .env` does not override a variable already set in the process: `$env:LLM_PROVIDER =
  "openai"` for one command works without editing `.env`.
- gpt-4.1-mini (Phase 1's model) shuts down 2027-01-14.
- Current progress: the OpenAI switch is built (this commit); live checks wait for the $5; then the image go.

**PLAN.md §3, the table:**
- LLM: "OpenAI gpt-5-mini (text) + gpt-5.1 (critic), $0 within the data-sharing free daily tokens; Gemini free
  tier kept behind `LLM_PROVIDER=gemini`".
- Embeddings: "gemini-embedding-2 @768, whatever `LLM_PROVIDER` says" (the "OpenAI's model stays behind
  `LLM_PROVIDER=openai`" note goes: it no longer follows the chat provider).
- Images: "OpenAI gpt-image-2, low ~$0.005 / medium ~$0.041 per 1024x1536 image (verified org needed); fal Nano
  Banana $0.0398 or FLUX ~$0.003 as settings": **~$0.03 per video at low** (6 images), ~$0.25 at medium.
- The cost-per-video line: images-only at low ≈ $0.03 ($5 ≈ 160 videos); with a hero clip ≈ $0.28-0.63.
