# FDA Warning Letter RAG + Chatbot

## Project Overview

A retrieval-augmented generation (RAG) system for querying FDA Warning
Letters. It ingests a corpus of letters into a vector database, retrieves the
most relevant passages for a user's question, and asks an LLM to answer using
only that retrieved context. A Streamlit chat UI sits on top so it can be used
interactively instead of through the terminal.

**Pipeline:**
1. Load FDA Warning Letter HTML files and strip everything except the letter
   body (`role="main"`), so repeated site boilerplate — nav bars, footers,
   "Content current as of..." — never gets embedded.
2. Split each letter into ~1000-character chunks (200-character overlap).
3. Embed each chunk (Qwen3-Embedding-8B via the Nebius Token Factory API,
   OpenAI-compatible) and upsert into a Pinecone index.
4. On a question: embed the question, retrieve the top 5 most similar chunks
   from Pinecone, and pass them as context to Gemini (`gemini-3.6-flash`) to
   generate a grounded answer.
5. `chatbot_app.py` wraps step 4 in a Streamlit chat interface with message
   history.

**Files:**
- `FDA warning Letter RAG` — ingestion pipeline + `generate_answer()` + a
  terminal chat loop (`python "FDA warning Letter RAG"`).
- `chatbot_app.py` — Streamlit chat UI (`streamlit run chatbot_app.py`) that
  reuses `generate_answer()` from the script above without modifying it.
- `requirements.txt` — pinned dependencies.
- `dotenv.env` (not committed) — holds `NEBIUS_API_KEY`, `PINECONE_API_KEY`,
  `PINECONE_INDEX_NAME`, and the Gemini credentials.

## Dataset

FDA Warning Letters (HTML), 2026, sourced from FDA's public warning letter
pages and stored locally under `fda_warning_letters/docs/2026/`. Only the
`role="main"` content block from each page is extracted — this is the actual
letter text, excluding the site chrome that's near-identical across every
letter and would otherwise dominate similarity search results. The corpus
produced **397 chunks** after splitting.

## Prompt Used

```
You are a regulatory intelligence analyst working with a corpus of FDA Warning
Letters. You answer questions about what FDA has cited, using only the letters
supplied to you in the CONTEXT block.

CONTEXT:
{context}

Question: {question}
Answer:
```

The system role ("regulatory intelligence analyst") and the "using only the
letters supplied" instruction are what keep answers grounded in the retrieved
context rather than the model's general knowledge.

## Iterations

- **Idempotent ingestion.** Early runs re-embedded and re-upserted the full
  corpus every time the script ran. Fixed by checking Pinecone's
  `describe_index_stats()` vector count before ingesting — if the namespace
  already has vectors, ingestion is skipped entirely.
- **Boilerplate filtering.** Initially the whole HTML page (including nav,
  footer, and "FDA Archive" links) was embedded. Since that text repeats
  almost verbatim across every letter, it was crowding out real content in
  similarity search. Fixed by extracting only the `role="main"` element with
  BeautifulSoup before chunking.
- **Rate limiting.** Bulk-embedding ~400 chunks against the Nebius API hit
  rate limits. Fixed with batches of 20 chunks, a retry loop with backoff
  (`20s * attempt`, up to 5 attempts) on `RateLimitError`, and a 2-second
  pause between batches.
- **Gemini response format.** Newer Gemini models return `response.content`
  as a list of content blocks (text + signature metadata) instead of a plain
  string. `generate_answer()` handles both shapes.
- **Wrapping the script in a chatbot without editing it.** The RAG script has
  no `.py` extension (it's meant to be run directly), so `chatbot_app.py`
  can't just `import` it normally.
  - First attempt used `importlib.util.spec_from_file_location`, which
    infers a loader from the file's extension — with no recognized extension
    it silently returned `None`, causing
    `AttributeError: 'NoneType' object has no attribute 'loader'`.
  - Fixed by explicitly constructing an `importlib.machinery.SourceFileLoader`
    and passing it to `spec_from_loader`, bypassing extension-based
    inference.
  - Loading it this way also gives the module a name other than `"__main__"`,
    so the original script's terminal input loop never fires when it's
    imported by Streamlit.
- **Avoiding repeated setup cost.** The RAG pipeline's setup (heavy imports,
  Pinecone connection, LLM client) took ~85 seconds. Wrapped the loader in
  `@st.cache_resource` so it only runs once per server process instead of on
  every chat message or script rerun.

## Learnings / Observations

- A vector-count check against the index is a simple, file-free way to make
  ingestion idempotent — no separate "already ingested" flag or database
  needed.
- When scraping many pages from the same site template, filtering to the
  actual content region before chunking matters more than chunk size or
  overlap tuning — boilerplate noise directly degrades retrieval quality.
- Embedding APIs' per-minute rate limits are easy to hit with only a few
  hundred chunks; batching + backoff should be assumed necessary, not added
  reactively.
- `importlib.util.spec_from_file_location` fails silently (returns `None`)
  for files without a recognized extension rather than raising — worth
  knowing before debugging the `NoneType` error it produces downstream.
- In Streamlit apps, anything expensive in setup (SDK imports, network
  handshakes) needs `@st.cache_resource` or it silently re-runs on every
  interaction. What looks like "re-ingesting on every run" can just be
  cold-start overhead being paid once per process — worth distinguishing
  before treating it as a bug.
