#### Ingestion pipeline flow
```
RSS feeds (feedparser)
        ↓
  fetch full text (trafilatura)
        ↓
  content hash check (skip if exists)
        ↓
  insert into articles (status: pending)
        ↓
  generate summary (DeepSeek)
        ↓
  chunk text (SentenceSplitter)
        ↓
  embed all nodes (Voyage)
        ↓
  insert into article_nodes
        ↓
  update articles (status: chunked)
```
---

#### Ingestion phase folder 
```
TechnoRAG/
      ingestion/
            __init__.py
            feeds.py        # feedparser logic, returns raw feed entries
            fetcher.py      # trafilatura, returns clean text + metadata
            hasher.py       # content hash, dedup check
            chunker.py      # SentenceSplitter, returns nodes
            embedder.py     # Voyage API calls
            writer.py       # inserts into postgres
            pipeline.py     # orchestrates all of the above
```
---

#### Query phase pipeline
```
User Query (raw string)
        │
        ▼
[1. QUERY CLASSIFICATION]
Classify the query into one of four types:
  - temporal   → "what did Anthropic announce this week?"
  - conceptual → "how does constitutional AI work?"
  - comparative → "compare GPT-4o vs Claude 3.5"
  - factual    → "when did OpenAI release o3?"
This classification drives every parameter downstream.
Sets: top_k, dense/BM25 weight ratio, temporal threshold, prompt template.
        │
        ▼
[2. QUERY EMBEDDING]
Embed the raw query using voyage-4-nano.
model.encode(query, normalize_embeddings=True)
This is your dense retrieval vector.
        │
        ▼
[3. HYBRID RETRIEVAL]
Fire two searches simultaneously against article_nodes:

  Dense search  → pgvector cosine similarity on embedding column
                  returns top_k candidates with similarity scores

  Sparse search → Postgres full-text search on sparse_tsv column
                  using to_tsquery(), returns top_k candidates with BM25 scores

top_k is not fixed — it's set by the query classifier.
temporal queries → higher top_k (need more candidates to filter by date)
factual queries  → lower top_k (precision matters more than recall)
        │
        ▼
[4. RRF FUSION]
Reciprocal Rank Fusion merges the two ranked lists into one.
No score calibration needed — RRF works on ranks not raw scores.
Formula: score = 1/(k + rank_dense) + 1/(k + rank_sparse), k=60 standard.
Produces a single unified ranked list of candidates.
        │
        ▼
[5. RERANKING]
Take top N from RRF list, rerank using a cross-encoder.
Cross-encoder scores true query-document relevance — 
much more accurate than embedding similarity alone.
Why: embedding retrieval is fast but approximate.
Reranker is slow but precise. You run it on a small candidate set, not the whole corpus.
Use: voyageai rerank API or a local cross-encoder model.
Produces final ordered list of most relevant chunks.
        │
        ▼
[6. MMR — Maximal Marginal Relevance]
Remove redundancy from the reranked list.
MMR balances relevance vs diversity:
  score = λ × relevance - (1-λ) × max_similarity_to_already_selected
Prevents the LLM seeing 5 chunks that all say the same thing.
Produces diverse, non-redundant final context set.
        │
        ▼
[7. TEMPORAL ASSESSMENT]
For each chunk in the final context set, check published_at in metadata.
Compute a freshness score based on query type:
  temporal queries  → strict threshold, flag anything older than 7 days
  conceptual queries → loose threshold, older content is fine
  factual queries   → medium threshold
If critical chunks are stale, either:
  - warn the user ("based on content from X days ago")
  - filter them out entirely if below minimum freshness threshold
        │
        ▼
[8. GENERATION]
Build a prompt using the query type's template.
Each query type gets a different prompt:
  temporal   → emphasise recency, cite dates explicitly
  conceptual → explain clearly, depth over brevity
  comparative → structured comparison, balanced
  factual    → precise, direct, single answer
Feed: prompt + final chunks as context → DeepSeek
DeepSeek generates the answer with inline citation markers.
        │
        ▼
[9. FAITHFULNESS JUDGE]
Before returning to user, verify the answer against source chunks.
Send to DeepSeek (or a dedicated judge prompt):
  "Given these source chunks, does this answer contain any claims
   not supported by the sources? Flag each unsupported claim."
If hallucinations detected:
  - strip or flag the unsupported claims
  - optionally regenerate
If clean: pass through.
        │
        ▼
[10. RESPONSE]
Return to user:
  - final answer
  - citations (url, title, published_at per chunk used)
  - freshness metadata ("sources from last 3 days")
  - confidence signal (if faithfulness judge flagged anything)
```

---

#### Query phase folder
```
TechnoRAG/
      query/
            classifier.py     # query classification → type + parameters
            embedder.py       # query embedding (reuse voyage-4-nano)
            retriever.py      # dense + sparse search against postgres
            fusion.py         # RRF merging
            reranker.py       # cross-encoder reranking
            mmr.py            # diversity filtering
            temporal.py       # freshness scoring + staleness flagging
            generator.py      # prompt templates + DeepSeek call
            faithfulness.py   # hallucination judge
            pipeline.py       # orchestrates all of the above
```
---