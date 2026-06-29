from dataclasses import dataclass
from datetime import datetime, timezone
import json

import trafilatura
from dateutil import parser as dateutil_parser

from ingestion.feeds import FeedEntry, fetch_all_feed, FEEDS, HEADERS
from configurables.config import MAX_ARTICLES, logger

@dataclass
class FetchedArticle:
    src:            str
    url:            str
    title:          str
    text:           str
    published_at:   datetime | None
    author:         str | None

def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    
    try:
        dt = dateutil_parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt

    except Exception as e:
        print(e)

def _resolve_date(trafilutura_date: datetime | None, feed_date: datetime | None):
    return trafilutura_date if trafilutura_date is not None else feed_date

def article_fetcher(entry: FeedEntry) -> FetchedArticle:
    try:
        raw_html = trafilatura.fetch_url(entry.url)

        if raw_html is None:
            logger.info("Failed to fetch url")
            return None
        
        logger.info("fetched %s", entry.src)
        
        # output_format json gives us text + metadata in one pass
        result_json = trafilatura.extract(
            raw_html,
            output_format="json",
            with_metadata=True,
            include_comments=False,   # never want comment sections
            include_tables=False,     # flat prose only, no table noise
            favor_precision=True,     # prefer less text over noisy text
        )

        if result_json is None:
            logger.info("Extraction failed for %s", entry.url)

            return None

        result = json.loads(result_json)
        text = result.get("text", "").strip()

        if not text:
            logger.info("No body text found for %s", entry.url)
            return None
        
        # resolve date
        trafilatura_date = _parse_date(result.get("date"))
        published_at = _resolve_date(trafilutura_date=trafilatura_date, feed_date=entry.published_at)

        # prefer trafilutra title
        title = result.get("title") or entry.title

        fetched_entry = FetchedArticle(
            src=entry.src,
            url=entry.url,
            title=title.strip(),
            text=text,
            published_at=published_at,
            author=result.get("author"),
        )

        with open("./data/fetched_articles.jsonl", "a", encoding="utf-8") as f:
            f.write(f"{fetched_entry}")
            f.write("\n")

        return fetched_entry

    except Exception as e:
        print(e)

if __name__=="__main__":
    feed_entries = fetch_all_feed(FEEDS, headers=HEADERS)
    print(len(feed_entries))

    article_entries = []
    for entry in feed_entries:
        if len(article_entries) >=MAX_ARTICLES:
            break
        result = article_fetcher(entry)
        article_entries.append(result)

    # print(article_entries)