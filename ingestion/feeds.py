from dataclasses import dataclass
from datetime import datetime, timezone
import logging

import feedparser

from configurables.config import logger

FEEDS: dict[str, str] = {
    "anthropic":            "https://www.anthropic.com/rss.xml",
    "openai":               "https://openai.com/blog/rss.xml",
    "google_deepmind":      "https://deepmind.google/blog/rss.xml",
    "meta_ai":              "https://ai.meta.com/blog/rss/",
    "microsoft_research":   "https://www.microsoft.com/en-us/research/blog/feed/",
    "mistral":              "https://mistral.ai/news/rss",
    "huggingface":          "https://huggingface.co/blog/feed.xml",
    "techcrunch_ai":        "https://techcrunch.com/category/artificial-intelligence/feed/",
    "venturebeat_ai":       "https://venturebeat.com/category/ai/feed/",
    "the_verge_ai":         "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
    "wired_ai":             "https://www.wired.com/feed/tag/ai/latest/rss",
    "mit_tech_review":      "https://www.technologyreview.com/feed/",
}

HEADERS: dict[str, str] = {
    "User-Agent" : "TechnoRAG/1.0 (ingestion)"
}

@dataclass
class FeedEntry:
    src:            str
    url:            str
    title:          str
    published_at:   datetime | None
    summary:        str

def _parse_date(entry: dict[str, str]) -> datetime:
    try:
        struct = entry.get("published_parsed")
        if struct is None:
            return None

        return datetime(*struct[:6], tzinfo=timezone.utc) 
    
    except Exception as e:
        print(e)

def fetch_feed(source: str, url: str, headers: dict[str, str]) -> list[FeedEntry]:
    try:
        parsed = feedparser.parse(url, request_headers=headers)

        if not parsed.entries:
            logger.info("no entries found for %s", source)
            return []

        if parsed.bozo:
            logger.info("Malformed entries for %s: %s", source, parsed.bozo_exception)
        
        entries = []
        for entry in parsed.entries:
            _url = entry.get("link")
            title = entry.get("title")

            if not _url or not title:
                logger.info("Missing url or title, skipping entry %d from %s", len(entries), source)
            
            entries.append(FeedEntry(
                src=source,
                url=_url.strip(),
                title=title,
                published_at=_parse_date(entry),
                summary=entry.get("summary")
            ))
            
        return entries
        
    except Exception as e:
        print(e)

def fetch_all_feed(feeds_dict: dict[str, str], headers: dict[str, str]) -> list[FeedEntry]:
    all_entries = []
    for source, url in feeds_dict.items():
        entries = fetch_feed(source=source, url=url, headers=headers)
        all_entries.extend(entries)

        with open("./data/feedparser_data.jsonl", "a", encoding='utf-8') as f:
            f.write(f"{entries}\n")

    return all_entries

if __name__=="__main__":
    result = fetch_all_feed(FEEDS, HEADERS)