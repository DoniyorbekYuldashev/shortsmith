# Phase 5 — Research + RAG: scripts grounded in evidence, not guesses

> Read at your own pace, ask anything. Parts 0–11 are the concepts. Parts 12+ (our code, real YouTube
> results, real retrieval scores, the with/without experiment, errors, tests) are completed from the
> build session's notes once the build is done.

---

## 0. The problem: the Writer only knows what it learned in training

Remember the Step 1.4 scripts: a **pumpkin-spice latte** and a **fireplace** in a Tashkent café. The model
wasn't wrong, it was **generic**. It writes from its general knowledge, which is:
- **frozen** at its training date, so it doesn't know what's popular *this month*;
- **average**: it knows "café videos" in general, not *proven hook patterns* or *your brand's rules*.

Phase 5 gives the Writer **evidence** before it writes:
1. **What's working right now:** the most-viewed short videos on the topic from the last ~30 days (YouTube).
2. **What's proven to work:** a library of hook patterns and brand guidelines, of which only the few that
   fit this topic are found and added (**RAG**).

The Researcher turns both into short **research notes**, and the Writer writes with them. That's called
**grounding**: the output is anchored to real data instead of the model's guesses.

```
topic ─► YouTube search (trends) ─┐
                                  ├─► Researcher ─► short notes ─► Writer ─► Script
topic ─► RAG (hooks, brand) ──────┘
```

---

## 1. Trend research with the YouTube Data API

**What we ask YouTube** (`search.list`):
- `q` = the topic, `type=video`, **`videoDuration=short`** (under 4 minutes), **`order=viewCount`** (most
  viewed first), **`publishedAfter`** = 30 days ago;
- plus a language/region hint (`relevanceLanguage`, `regionCode`) so an Uzbek topic finds Uzbek videos.

Then **`videos.list`** for the exact **view counts and durations** of those videos (search results don't
include them), keeping real Shorts-length videos.

**Quota, which is the budget here:** 10,000 free units per day. **One search = 100 units**, a
`videos.list` call = 1 unit. That's about 100 searches a day. Two habits protect it:
- **Cache:** save each topic's results on disk for ~24 hours, so re-running the same topic costs 0 units.
- **Count:** keep a running total of units used today (like the cost ledger, but for quota).

**Honest limits:**
- **Views ≠ quality.** A video can be big because of a famous channel, not because of its hook.
- **Search ranking is YouTube's choice**, and some topics have few recent Shorts.
- **"Short" means under 4 minutes**, so some results aren't real Shorts and have to be filtered by duration.
- **The API's terms** allow short-lived caching for our own use, not republishing the data.

So trends are **signals**, not truth. That's why the Researcher summarises them instead of copying them.

---

## 2. RAG: Retrieval-Augmented Generation

**RAG = before generating, *retrieve* the relevant pieces of your own knowledge and put them in the
prompt.** The model then writes with that knowledge in front of it. Analogy: an **open-book exam**. The
student (the model) doesn't memorise the whole library; a librarian (retrieval) brings the 5 pages that
matter for *this* question.

RAG has two separate phases:
```
INDEXING (once, offline)                         QUERYING (every request)
────────────────────────                         ────────────────────────
documents (data/hooks/)                          the topic / question
   │ chunk: split into pieces                       │ embed: turn it into a vector
   ▼                                                ▼
chunks ─► embed each one ─► vectors ─► store      find the nearest vectors in the store
                                       (Chroma)     │ top-k: the k closest chunks
                                                    ▼
                                                  add those chunks to the prompt ─► LLM writes
```

---

## 3. Embeddings: coordinates for meaning

An **embedding** turns a piece of text into a **list of numbers** (a vector), often hundreds or thousands
long. It's made by a special **embedding model** (not a chat model) trained so that:

> **texts with similar meaning get vectors that point in similar directions.**

Analogy: **GPS coordinates for meaning.** "Start with a surprising question" and "open with a question
nobody expects" land close together. "Use warm golden lighting" lands far away.

- Numbers alone mean nothing; only **distances between vectors** mean something.
- A **multilingual** model puts the same idea in different languages close together: an Uzbek sentence
  about "savol bilan boshlash" (starting with a question) lands near the English one. That matters for us.
- Embedding is **cheap**: you pay only for input tokens, and on Gemini's free tier, nothing.
- **Everything must be embedded with the same model.** Vectors from two different models are like
  coordinates from two different maps.

---

## 4. Cosine similarity: how "close" is measured

Two vectors are compared by the **angle** between them:
```
cosine_similarity(a, b) = (a · b) / (|a| × |b|)
     a · b  = a1*b1 + a2*b2 + ...        (dot product)
     |a|    = sqrt(a1² + a2² + ...)      (length)
```
- **1.0** = same direction (same meaning), **0** = unrelated, **−1** = opposite.
- Why the **angle** and not the plain distance? The **direction** carries the meaning; the **length** often
  just reflects how long or wordy the text is. Cosine ignores length.

A tiny 2-D example (real embeddings have thousands of dimensions, but the maths is the same):
```
query            q = (0.9, 0.1)
"question hook"  a = (0.8, 0.2)   → cos ≈ 0.99   very close
"lighting tip"   b = (0.1, 0.9)   → cos ≈ 0.22   far
```
You'll write `cosine_similarity` and `top_k` yourself in the exercise. They're a few lines each.

---

## 5. The vector database (Chroma)

A **vector database** stores, for every chunk, its **vector**, its **text** and **metadata** (file, pattern
name, language…), and answers "give me the k nearest vectors to this one" **fast**.
- **Chroma** runs locally, as a library, with no server. It saves everything in a folder (`chroma/`,
  gitignored, because it can always be rebuilt from `data/hooks/`).
- For speed it uses an **approximate nearest-neighbour index** (HNSW), which avoids comparing the query
  with every single vector. With 100 chunks that doesn't matter; with millions it does.
- We give Chroma **our own embeddings** (from Gemini), instead of letting it download its own small model.
- **Indexing must be idempotent:** re-running it must not create duplicates, and a changed file must be
  re-embedded.

---

## 6. Chunking: how big should a piece be?

A **chunk** is the unit you embed and retrieve. Its size matters a lot:

| Chunks too **big** (e.g. the whole hooks file) | Chunks too **small** (e.g. one sentence) |
|---|---|
| one vector mixes many ideas, so it matches everything a little and nothing well (a "blurry" vector) | a piece loses its context: the template ends up separated from its "when to use it" |
| retrieving it pulls lots of irrelevant text into the prompt (wasted tokens) | you need many more chunks to get the full idea; retrieval gets noisy |

Common strategies:
- **fixed size** (e.g. ~500 tokens, with some **overlap** so sentences cut at a border appear in both pieces);
- **by structure**: one natural unit per chunk (a section, a paragraph, a record).

**Our choice: by structure.** One hook pattern = one chunk (name, template, examples, when to use, why it
works), and brand guidelines are split by section (tone, words to avoid, visual style). Each chunk is
one complete idea.

---

## 7. top-k: how many chunks to retrieve

- **k too small** (1–2): you may miss a better pattern that scored slightly lower.
- **k too large** (20+): more noise and more tokens, and the useful ones get diluted.
- A typical start is **k = 3–8**, sometimes with a **minimum score** so weak matches are dropped.
- The scores also help you **debug**: if the best match only scores 0.3, the knowledge base probably has
  nothing relevant for that topic.

---

## 8. Why not just put all 100 hooks in the prompt?

The plan's check question. Four reasons:
1. **Tokens on every call.** 100 patterns could be several thousand tokens, sent with *every* script
   request (Phase 1: every token in the prompt is paid each time). The build will measure the real number.
2. **"Lost in the middle."** Models pay less attention to the middle of long prompts. Five relevant patterns
   get more attention than one hundred.
3. **Noise.** 95 irrelevant patterns can pull the script in random directions ("POV hook" for a recipe
   video?). Retrieval hands over only what fits *this* topic.
4. **It doesn't scale.** 100 patterns might fit; 10,000 brand rules, product facts and past scripts won't.
   RAG works the same at any size.

---

## 9. RAG vs fine-tuning vs long context

| | **RAG** | **Fine-tuning** | **Long context** |
|---|---|---|---|
| What changes | what's **in the prompt**, chosen per request | the **model's weights** | the prompt gets very big |
| Adding new knowledge | edit a file, re-index (minutes) | collect examples, retrain (hours, costs money) | paste it in |
| Cost per call | small (a few chunks) | none extra | large (all of it, every call) |
| Best for | facts that change, private data, citing sources | a **style, format or skill** (e.g. "always write Uzbek SQL like this") | a few documents read once |
| Shortsmith | ✅ hooks, brand rules, trends | ❌ not needed here | ❌ wasteful for 100 patterns |

**Rule of thumb:** RAG adds **knowledge**; fine-tuning changes **behaviour**. Your second project (Uzbek
text-to-SQL) is a good case for **fine-tuning**: teaching a small model a skill and a style. Shortsmith's
hooks are a good case for **RAG**: knowledge you'll keep editing.

---

## 10. Research notes: context engineering

The Researcher doesn't hand the Writer raw data (25 video titles, 5 full hook patterns, a brand guide).
It writes **short structured notes**:
- what's trending;
- which 2–3 hook patterns fit, and why;
- angles to try;
- things to avoid.

Why: the Writer gets **only what helps**, in **few tokens**. That's **context engineering** (Phase 6.9 in
full): every agent receives exactly what it needs and nothing more. The notes are a Pydantic model, so the
Writer always gets the same shape.

---

## 11. Evaluation: did research actually help?

"It feels better" isn't evidence. The experiment (Step 5.5) is a **side-by-side**:
- the same topics, written **with** and **without** research, with the same model and settings;
- recorded for each: the hook, which retrieved pattern the script really used, and a score;
- scores from an **LLM-as-judge** (a model scoring against a fixed rubric) **and** a human reading;
- **honest caveats:** a few topics prove little (Phase 1.5's lesson), and a judge model has its own biases.
  This is a demonstration, not statistics.

The build notes will show the real results, including where research *didn't* help.

---

## 12. Our code

### 12.1 `research/youtube.py`: trends
- **Two requests per live search:**
  - `search.list` (100 units): topic, `type=video`, `order=viewCount`, `videoDuration=short`,
    `publishedAfter` = 30 days ago, a language hint, and `regionCode=UZ` for Uzbek/Russian;
  - then `videos.list` (1 unit, up to 50 videos) for views, likes, exact length.
- **Filtering:** the length comes as ISO 8601 (`"PT1M2S"` = 62 s). It keeps videos of **≤ 180 s** (Shorts can
  be 3 minutes since October 2024), sorts by views, and keeps the top 10.
- **"Is it vertical?"** The API has no "is a Short" field. The trick: ask for the embed **player** at a
  small width, and its size follows the video's shape. A real Short came back 180 × 320 (9:16).
- **Pydantic models:** `TrendVideo` (title, channel, views, published, duration, vertical, url) and `TrendReport`.
- **The key travels in the `X-Goog-Api-Key` header**, never in a URL, log line or error (a test checks it).
- **Cache:** `.cache/youtube/<hash of the query>.json`, fresh for **24 hours**. Every search first deletes
  **all** expired files: YouTube's terms allow keeping data only briefly. Case and extra spaces don't
  matter: `"  Coffee shop   TASHKENT "` hits the same file.
- **Quota log** `runs/youtube_quota.jsonl`, one line per request (refused ones too, since Google charges them).
  - The day is counted in **Pacific time**, because that's when Google resets the quota.
  - **Our own cap is 2,000 units a day** (Google allows 10,000), checked *before* any request, so a bug in
    a loop can't burn the whole quota.

### 12.2 `data/hooks/`: the knowledge base (committed)
- **69 original hook patterns** in 8 files (curiosity, numbers, contrast, story, practical, place,
  visual, emotion). **Each** has a template, examples in **English, Uzbek and Russian**, "when to use" and
  "why it works" (a test checks all four fields and all three languages).
- **Format: one `## ` section per pattern**, so chunking is just "split at `## `":
  ```markdown
  ## Open loop
  **Template:** "[Surprising fact]. Why? I'll show you at the end."
  **Examples:**
  - en: "This café has no menu. The reason is genius."
  - uz: "Bu kafeda menyu yo'q. Sababini oxirida aytaman."
  - ru: "В этом кафе нет меню. Причина вас удивит."
  **When to use:** a topic with one strange, concrete fact you can explain in the last scenes.
  **Why it works:** an unanswered question is uncomfortable; people stay to close it.
  ```
- **2 fictional brands:** *Anor va Dolchin* (a café in Mirobod, Tashkent) and *Samoviy Karvon* (Silk Road
  trips), each with About, Voice and tone, Words we use and avoid, Visual style, Content rules.

### 12.3 Embeddings: `llm.embed()`
- **Model: `gemini-embedding-2` at 768 numbers**, on the free tier, through the same provider switch as chat.
  **This replaces the plan's OpenAI embeddings** (your OpenAI account is at $0); OpenAI's model stays behind
  `LLM_PROVIDER=openai`.
- **Why that model**, from a probe with the same sentence in English and Uzbek:

  | Model | Size | Similarity en↔uz |
  |---|---|---|
  | gemini-embedding-001 | 768 | 0.83 (and not normalised) |
  | **gemini-embedding-2** | **768** | **0.90** |

  It's better across languages, which matters for Uzbek topics.
- **768 instead of the default 3072** is 4× less to store and compare, and measured to lose nothing (section 14).
- **"Task formats" in the text:** Google's guide says to embed documents as `title: Open loop | text: ...` and
  queries as `task: search result | query: ...`. The compatible endpoint rejects the `task_type` option,
  so the format goes into the text itself. Measured: a small improvement.
- **The ledger line is written right after the response, before any parsing** (bug E1). Gemini doesn't report
  token usage for embeddings, so tokens are **estimated** with tiktoken (±7%) and marked `tokens_estimated: true`.
- **Rate limit:** on the free tier **every text counts as one request, 100 per minute** (E2). A 429 waits as
  long as the error says, then retries.

### 12.4 `research/rag.py`: the index and retrieval
- **Chunking:** `split_sections()` → one chunk per `##` section. Ids are readable and stable:
  `curiosity/open-loop`, `brands/anor-va-dolchin/visual-style`. **79 chunks**: 69 patterns (91–134
  tokens, median 110) + 10 brand sections.
- **Chroma:** a persistent collection in `chroma/`, named per model and size (`kb-gemini-embedding-2-768`,
  because vectors from different models live in different "maps"). It uses **cosine**, and
  `embedding_function=None`: **Chroma stores, we embed**. It never downloads its own model, and every
  embedding goes through our ledger.
- **`sync()` is idempotent:** each chunk's text hash is stored with it. On a re-run, only new or edited
  chunks are embedded, deleted ones are removed, and unchanged ones cost nothing:
  ```
  first run:   +79 new | embedded 9370 tokens | 30.29 s (one rate-limit wait)
  second run:  79 unchanged | embedded 0 tokens | 0.02 s
  ```
  It's so cheap that it runs **before every retrieval**, so the index can never be out of date.
- **`retrieve(query, k=5)`** returns the k most similar chunks with **similarity = 1 − cosine distance**.
- **`brand_rules(topic, brand)`** asks for "*topic: tone, words to use and avoid, visual style*", because the
  bare topic missed the brand's hard rules (E7).
- **`import chromadb` happens only when needed** (it takes 2–9 s), so the MCP server still starts in 2 s.

### 12.5 The new tools, and a least-privilege fix
- The registry now has **two groups**:
  - **media** (`generate_image`, `text_to_speech`, `render_video`), which work inside a run;
  - **research**: **`search_trends(query, language)`** and **`retrieve_hooks(query, k, brand)`**.
- **They appear in the MCP server automatically**, so Claude Desktop can research too.
- **The producer agent still sees only the 3 media tools** (every extra schema is paid on every turn).
- **E10, found in review:** the producer was only *shown* 3 tools, but `call_tool()` would run *any*
  registry name. A model that invented a `search_trends` call would have spent YouTube quota. Fix:
  `offered_only()`. A tool that wasn't offered gets *"unknown tool"*. **What a model may do is decided by the
  program, not by what it asks for.**

### 12.6 `agents/researcher.py`, `write_script` and `make`
- **`research(topic, language, brand)` is code plus *one* LLM call, not an agent loop** (D9). What to search
  and retrieve is predictable from the topic, so plain code does it:
  1. YouTube with the topic's first part; if under 3 Shorts come back, once more with one word dropped;
  2. brand rules;
  3. the top 6 patterns.

  Then **one `chat_json` call** writes the notes, where the LLM does what code can't: filter noise,
  pick patterns, write. On the free tier every agent turn costs 7–300 s, so this saves real time.
- **`ResearchNotes`** (Pydantic): `trending`, `hooks` (pattern + hook + why), `angles`, `avoid`, `visual_style`.
  Long lists are **cut, not rejected** (a rejection would waste a paid call). `for_writer()` renders them as
  a few lines **with no scores, ids or URLs**.
- **`Research`**, saved as **`research.json`** in the run folder, keeps everything: the trend report, the
  retrieved patterns with scores, and the notes. That lets you see **why** a script came out the way it did.
- **`write_script(..., research=notes)`** puts the notes in the **user** message; the system prompt is unchanged.
  Without research, the message is byte-identical to Phase 1's.
- **`make "topic" --research --brand anor-va-dolchin`** runs it all end to end.
- **YouTube failing never stops a video** (D10): the notes then say "no trend data" and the hooks still come.

---

## 13. Real results

### YouTube: "coffee shop in Tashkent", live, 101 units
| Views | Length | Title (as on YouTube) |
|---:|---:|---|
| 266,504 | 46 s | В Ташкенте мальчик решил помочь семье и открыл небольшое самодельное кафе |
| 33,109 | 56 s | Chirchiqdagi Safia 👧🏻 Safia в Чирчике |
| 15,379 | 105 s | Uzbekistan🇺🇿 Finding an exceptional cup of coffee while traveling through Central Asia… |
| 13,168 | 59 s | 10-летний мальчик открыл кафе на улице в Ташкение |
| 6,254 | 68 s | Новая пекарня в Ташкенте🥐☕️ |

- The language hint is **a hint, not a filter**: 8 of the top 10 were Russian or Uzbek.
- **The top video is local news** (a boy opening a café), and café openings and bakeries get the views.

### Uzbek: two lessons about keyword search
- `"Samarqand Registoni kechqurun"`: **0 results** in 30 days. Too specific, hence the "drop one word" retry.
- `"Samarqand Registoni"`: 4 good results (a surprise music performance on the square had 326k views)…
  **and 6 Indian videos about the desert.** *"Registan" means desert in Hindi/Urdu.* **Keyword search knows
  words, not meanings.** The Researcher's prompt now says to ignore off-topic results, and it did.

**Quota:** 806 units used that day (8 searches), out of our 2,000 cap and Google's 10,000. Cached repeats cost 0.

### Retrieval, with real scores
```
top 5 for 'cozy coffee shop in Tashkent, autumn vibe':        top 5 for 'Toshkentdagi shinam qahvaxona, kuzgi kayfiyat':
  0.6703  POV                                                  0.6826  POV
  0.6583  The hidden gem                                       0.6346  Unpopular opinion
  0.6465  Save the address                                     0.6331  The hidden gem
  0.6443  Only this month                                      0.6289  Only this month
  0.6347  Walk with me                                         0.6278  Cheap vs expensive
```
- **It works across languages:** the Uzbek query finds almost the same patterns.
- **The scores are compressed** (0.67 → 0.63). Similarity is a **ranking signal, not a probability**: in a
  library of 69 hook patterns everything is "somewhat similar". A fixed threshold like "only above 0.7" would
  have returned nothing, so **top-k is the robust choice**.

### The Researcher: first a real failure, then a fix
**Version 1** made two mistakes, found by **reading the notes against the trend table**:
- it called the 266k video "a bakery video" (it's a boy's homemade café);
- it suggested the hook **"Save this address: Ukchi street, Tashkent"**: the address of **another
  business**, copied from a trend title, when our café is in Mirobod.

**Version 2 of the prompt:** *"describe a video by what its title says"* and *"the videos are evidence of what
viewers like, not facts for our script: never reuse their addresses, names or claims."*
**Lesson: retrieved and searched text is untrusted input**, just like a tool result. Grounding data can
mislead as well as inform.

**The notes the Writer read** (267 tokens, from ~1,850 the Researcher read, so **about 7× smaller**):
```
What gets views now:
- Stories about independent neighborhood spots and bakeries in Tashkent draw strong viewership, such as the 10-year-old opening a cafe with 13,168 views.
Hook patterns that fit. Base your hook on one of them (adapt it, don't copy it word for word):
- POV: "POV: you found Tashkent's coziest autumn cafe." (It puts the viewer directly into the autumn scene ...)
- Save the address: "Save this address: your autumn table is waiting." (...)
Angles to use:
- Focus on the sound of freshly brewed hot tea or coffee matching the crisp autumn air outside.
Avoid:
- the cheapest / the best in the world / luxury / VIP
End every visual_prompt with these style words: warm amber light, terracotta and pomegranate red, 35mm film look, cozy
```
**The script is visibly grounded:**
- the hook is the retrieved **POV** pattern;
- the CTA is **"Save the address"**;
- scenes 3–4 are the two angles;
- all 6 image prompts end with the brand's style words;
- none of the banned words appear.

**But:** the hook was copied **word for word** ("adapt, don't copy" was ignored), and the script had **60 words**
instead of 78–84, which made a 23.4-second video.

---

## 14. The experiment: with vs without research (5.5)

**Setup, fair by design:**
- 4 topics (2 English, 2 Uzbek), each written **with** and **without** research by the same writer with the same settings;
- **the judge:** a *stronger, different* model (gemini-3.8-flash), which reduces "self-preference";
- **it's blind:** it sees "A" and "B", never which one had research, and only the spoken text;
- the rubric scores hook, specificity, relevance and flow, 1–10 each;
- **run twice per topic with A and B swapped**, to detect "position bias" (preferring whichever comes first).

| Topic | With research | Without | Winner (both orders) |
|---|---:|---:|---|
| coffee shop in Tashkent (en) | 41 | **50** | **without**, both times |
| saving money as a student (en) | **52** | 41 | with, both times |
| Registan in the evening (uz) | **47** | 40 | with, both times |
| plov secrets at home (uz) | **49** | 36 | with, both times |
| **total** | **189 / 320** | 167 / 320 | **research: 6 of 8 verdicts** |

**The hooks, side by side:**
| Topic | With research | Without |
|---|---|---|
| coffee shop | "POV: you found Tashkent's coziest autumn cafe." | "Escape the Tashkent autumn chill today." |
| saving money | "Don't spend money on campus until you see this." | "Stop wasting money in college right now." |
| Registan | "Registonni kechqurun hech qachon bunday ko'rmagansiz." | "Registon kechqurun ertakga aylanadi." |
| plov | "Oshning siri go'shtda deb o'ylaysizmi? Yo'q." | "Mazali palov tayyorlash sirlarini bilasizmi?" |

**What it shows:**
- **Both orders agreed on every topic**, so there's no sign of position bias.
- **Almost all the gain is in the hook: +15 of the +22 points.** That's exactly where retrieval acts.
- **"Visibly uses retrieved patterns": yes, 4 of 4**, but 3 hooks were the notes' suggestion **word for word**,
  and the plov hook is the **Uzbek example from `data/hooks/curiosity.md`, unchanged**. The writer delegates
  the hook. At scale, two videos would open with the same line.
- **Real trends shaped content** in 3 topics (student discounts and no-spend days; the Registan music performance;
  the trending Khorezm plov).

**Where research hurt or didn't help (the honest list):**
- **The coffee shop lost.** The notes' *visual* angles ("warm lighting, steam rising") turned into narration
  that **describes pictures**: "Warm amber lights glow softly against the cool autumn air outside today."
  Nobody talks like that.
- **Word counts missed the target in both arms** (7 of 8 scripts).
- **Uzbek grammar slips in both arms**, which the judge didn't notice ("ertakga" should be "ertakka"; "sirdosh"
  means "confidant", not "secret"). **An LLM judge is not an Uzbek proofreader.**
- **The limits:** 4 topics, one sample each, one judge model whose rubric rewards exactly the curiosity hooks the
  patterns supply. **Evidence, not proof.**

### The chunking question, measured (retrieval evaluation)
16 hand-written queries that describe a pattern **without using its words**, e.g. *"show the finished dish
first and then how it was made"* → *End first*; *"Раньше всё стоило дешевле, помните?"* → *Nostalgia*.
| Setup | hit@1 | MRR | Tokens in the top 3 |
|---|---:|---:|---:|
| **one pattern per chunk, task formats, 768** (what we use) | **0.88** | **0.91** | 334 |
| same, without task formats | 0.81 | 0.88 | 335 |
| same, 3072 numbers | 0.88 | 0.92 | 334 |
| **whole files (too big)** | 0.62 | 0.77 | **3,046** |
| **template line only (too small)** | 0.81 | 0.88 | **76** |

**Reading the metrics:** *hit@1* is "was the very first result right?". *MRR* (mean reciprocal rank) is 1 when the
right answer is first, ½ when second, ⅓ when third…, averaged.
- **Too big:** one vector averages 9 ideas, so it's blurry. The first result was right only 62% of the time, and
  every hit drags ~1,000 tokens into the prompt (9× more, mostly irrelevant).
- **Too small:** ranking survives (each chunk still has the pattern's name), but what comes back is **76 tokens**
  with no examples, no "when to use", no "why". The Researcher couldn't judge fit from that. **Too small hurts the
  *generation* more than the ranking.**
- **The rule:** a chunk is the **smallest piece that is useful on its own**. Here that's one pattern.
- **768 vs 3072:** the same quality at a quarter of the storage. **Task formats:** a small, free gain, kept.
- (With 16 queries, one query = 0.06, so differences of one query are noise-sized.)

### The plan's question, answered with numbers: why not put all the hooks in the prompt?
**All 69 patterns = 8,535 tokens. The 6 the Researcher gets = 779 tokens (11× less)**, on every call. Plus
attention (6 relevant options are easier than 69), latency, and scale (1,000 patterns or a 50-page brand guide
wouldn't fit).
**Honest counterpoint:** 8.5k tokens *would* fit easily in a 1M-token context window. At this library size, long
context is a legitimate design too. RAG wins here mainly because it also keeps the **Writer's** prompt short
(267 tokens of notes, not 8.5k).

---

## 15. Errors that happened, and what they teach
| # | What happened | Lesson |
|---|---|---|
| E1 | `TypeError … 'int' and 'NoneType'`: Gemini sends `index: null` on embeddings (OpenAI numbers them), **and a served request never reached the ledger** because the code parsed before recording | **record the ledger line right after the response**, before anything that can fail; compatible ≠ identical |
| E2 | `429 RESOURCE_EXHAUSTED … embed_content_free_tier_requests, limit: 100` | on the free tier **each text is one request**: batching saves round trips, not quota. Wait the stated time and retry |
| E3 | a cached search was deleted after the cache format changed, costing one more search | changing a cache's format invalidates the cache (correct behaviour, just costly) |
| E4 | a natural Uzbek topic found 0 trends | specific queries can find nothing, so add a broader fallback |
| E5 | "Registan" returned Indian desert videos | **keyword search knows words, not meanings**: filter for relevance |
| E6 | the Researcher distorted a video and **reused another business's address** | **retrieved and searched text is untrusted input**; tell the model what it may and may not reuse |
| E7 | brand retrieval missed "Words we use and avoid" | rules that must *always* apply shouldn't depend on similarity (open item: always include them) |
| E8 | an unknown brand still spent an embedding request | check cheap things before paying for expensive ones |
| E9 | a test fake "forgot" its answers (a new fake was built on every call) | build a fake once, return the same object |
| E10 | the producer could run tools it was never shown | **least privilege:** only offered tools may run |
| E11 | the build session **opened `.env` once** to list variable names (values were masked; no key was shown or stored) | an honest process slip: the rule "never open `.env`" exists because "just the names" is still opening it |

**Open items** (for later phases):
- the writer misses its word targets with or without research;
- split the notes' "angles" into *what to say* and *what to show*;
- stop the writer copying hooks word for word;
- always include a brand's "words to avoid";
- filter trend titles by embedding similarity in code;
- better search queries for Uzbek topics.

---

## 16. Tests, and your exercise 🧠

- **235 tests** (+81): test_youtube 23, test_rag 20, test_researcher 18, and more in llm/tools/mcp/loop.
  - **YouTube:** faked with a `MockTransport` (no network). Checked: the key in the header and not the URL, the
    Shorts filter, cache hits for 0 units, the **Pacific-time quota day**, our cap refusing before any request,
    errors never cached and never containing the key.
  - **RAG:** bag-of-words **fake embeddings** + a **real Chroma** in a temp folder. Checked:
    - the real knowledge base's format (4 fields × 3 languages);
    - **a second sync embeds nothing**, and an edited section is re-embedded alone;
    - **cosine ignores vector length** (mutation-checked: with plain distance the test fails);
    - an unknown brand costs no request.
  - **Researcher:** exactly one structured LLM call; the prompt holds the trends, patterns and brand rules **with no
    URLs or scores**; YouTube failure handled; `make --brand` passes the notes.
- **🧠 Your exercise:** `exercises/phase-5/rag_math.py`, the whole idea of RAG in plain Python with no API and no Chroma:
  1. `split_sections(markdown)`: chunking;
  2. `cosine_similarity(a, b)`: what Chroma computes;
  3. `top_k(query, vectors, k)`: retrieval;
  4. `plan_sync(stored, current)`: why re-indexing is idempotent.

  The tests use tiny 2–3-number vectors, so you can **check every answer on paper**:
  ```
  uv run pytest exercises/phase-5
  ```
  14 tests, all red now. This one is short and very interview-relevant.

---

## Check questions (optional, answer whenever you like)
1. Why are the Writer's scripts generic without research? What two kinds of evidence does Phase 5 add?
2. Why do we cache YouTube results and count quota units? What does one search cost?
3. What are the two phases of RAG, and what happens in each?
4. What is an embedding? Why must every chunk and query be embedded with the same model?
5. What does cosine similarity measure, and why the angle instead of the distance?
6. What goes wrong when chunks are too big? Too small? How did we chunk the hooks?
7. Why not put all 100 hooks in the prompt? Give at least three reasons.
8. RAG vs fine-tuning: which adds knowledge, which changes behaviour? Which fits Shortsmith, which fits your Uzbek text-to-SQL project?
9. Why does the Researcher write short notes instead of passing raw data to the Writer?
10. How would you prove research made the scripts better, and what are the limits of that proof?

**Our build**
11. Why does the YouTube key travel in a header, and why does every search delete *all* expired cache files?
12. Why is our own quota cap 2,000 when Google allows 10,000? Why count the day in Pacific time?
13. Why did we pick gemini-embedding-2 at 768 numbers? What did the probe and the evaluation show?
14. What makes `sync()` idempotent, and why can it run before every retrieval?
15. Why "Chroma stores, we embed" (`embedding_function=None`)?
16. Why are similarity scores like 0.63–0.67 fine, and why top-k instead of a threshold?
17. The Researcher v1 reused another business's address. What's the general lesson?
18. Why is the Researcher code + one LLM call instead of an agent loop?
19. In the experiment, where did research help most, where did it hurt, and why?
20. What did the chunking evaluation show for "too big" and "too small"?

## Interview angles
- "RAG grounds generation in my own knowledge: I chunk by structure, embed with a multilingual model,
  store in Chroma and retrieve the top-k by cosine similarity, instead of stuffing everything into the prompt."
- "RAG adds knowledge, fine-tuning changes behaviour: hooks and brand rules change often, so they live in
  a retrievable knowledge base, not in the model's weights."
- "Trend research uses the YouTube Data API within its free quota, with caching and a unit counter, and the
  Researcher compresses trends and retrieved patterns into short structured notes for the Writer."
- "I evaluated research with a side-by-side experiment and an LLM judge plus my own reading, and reported
  where it didn't help."
- "Research won 6 of 8 blind verdicts, judged in both orders by a stronger model; almost all the gain was in
  the hook, and it lost one topic where visual notes leaked into the narration."
- "I measured chunking instead of guessing: one-pattern chunks hit 0.88 at rank 1, whole files 0.62 with 9×
  the tokens, and template-only chunks ranked fine but were useless as context."
- "Retrieved and searched text is untrusted input: my Researcher once copied another business's address
  from a trend title, so the prompt now separates evidence of what viewers like from facts we may use."
