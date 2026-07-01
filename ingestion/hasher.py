"""
1. Hashing:
- hash of the text instead of title/url: avoids content change
- hexdigest(): clean 64 char hex string (TEXT(64))

2. Duplicates checking:
- check via existing dict, which is loaded in the start
- Three cases:
    1. if url is not seen, ingest
    2. if url and hash identical, skip
    3. if url same but hash different, content updated, re-ingest
"""
import hashlib

from ingestion.fetcher import FetchedArticle
from configurables.config import logger

def hash_article(article: FetchedArticle) -> str:
    normalised = article.text.lower().strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

def is_duplicate(url, content_hash, existing_dicts):
    if existing_dicts:
        if url not in existing_dicts:
            return False # new article

        if existing_dicts[url] == content_hash:
            logger.info("existing, Skipped %s", url)
            return True # existing
        
        logger.info("Updated article, re-ingesting %s", url)
        return False 
    
    return False
