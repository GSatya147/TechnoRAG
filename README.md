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
#### Ingestion folder 
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