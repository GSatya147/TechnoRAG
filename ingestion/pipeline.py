from configurables.config import logger, MAX_ARTICLES_TESTING

from client.client import Client
from ingestion.chunker import summary_node, chunk_article
from ingestion.embedder import Embedder
from ingestion.fetcher import article_fetcher
from ingestion.feeds import fetch_all_feed, FEEDS, HEADERS
from ingestion.hasher import hash_article, is_duplicate
from ingestion.writer import insert_articles, insert_nodes, update_article_status, get_db, clear_db, create_db
    
def build_existing_dict(article_table):
    existing_dicts: dict = {}
    for row in article_table:
        if row:
            existing_dicts["url"] = row.get("url")
            existing_dicts["content_hash"] = row.get("content_hash")
        else:
            return None
        
        return existing_dicts

def build_article_entries(FEEDS, HEADERS, MAX_ARTICLES):
    feed_entries = fetch_all_feed(FEEDS, headers=HEADERS)

    article_entries = []
    for entry in feed_entries:
        if len(article_entries) >=MAX_ARTICLES:
            break

        result = article_fetcher(entry)
        article_entries.append(result)
    
    return article_entries

def chunking(article_entries, existing_dict):
    client = Client()
    chunk_nodes: list[dict] = []
    nodes_list: list[dict] = []

    for entry in article_entries:
        logger.info("Executing chunking on article: %s", entry.url)

        article_hash = hash_article(entry)

        status = is_duplicate(entry.url, article_hash, existing_dicts=existing_dict)

        if status:
            logger.info("Duplicate found, skipping")
            continue
       
        logger.info("No duplicate found, inserting")
        article_id = insert_articles(article=entry, content_hash=article_hash)

        # summary building
        context = [{"role" : "system", "content" : "You are a summary provder. Provide crisp and comprehensive summary well under 750 words"}, {"role" : "user", "content" : entry.text}]
        summary_text = client.generate_response(messages_list=context)
        # summary_text = f"This is summary for {entry.url}"
        summary = summary_node(article=entry, summary_text=summary_text, article_id=article_id)

        # chunk nodes
        chunk_nodes += chunk_article(article=entry, article_id=article_id)

        # update the ingestion status
        if chunk_nodes:
            update_article_status(article_id=article_id, status="chunked")
        else: 
            update_article_status(article_id=article_id, status="failed")

        # building nodes list to insert into postgres
        nodes_list += summary + chunk_nodes

    return nodes_list

def embedding(nodes_list):
    embedding_obj = Embedder()
    modified_nodes = embedding_obj.embed_and_modify(nodes_list)

    if modified_nodes:
        return modified_nodes
    
    return None

def ingestion_pipeline_run():
    article_table, article_nodes_table = get_db()

    existing_dict = build_existing_dict(article_table=article_table)
    entries = build_article_entries(FEEDS=FEEDS, HEADERS=HEADERS, MAX_ARTICLES=MAX_ARTICLES_TESTING)

    nodes = chunking(entries, existing_dict=existing_dict)
    logger.info("nodes: %s", len(nodes))
    logger.info("nodes: %s", nodes)

    modified_nodes = embedding(nodes_list=nodes)

    if modified_nodes:
        insert_nodes(modified_nodes)
        updated_article_table, updated_article_nodes_table = get_db()
        logger.info("article table: %s\narticle nodes: %s", updated_article_table, updated_article_nodes_table)

    else:
        logger.info("Embeddings are none..")

if __name__=="__main__":
    ingestion_pipeline_run()

    

















