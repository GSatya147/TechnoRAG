import json

from db.db_client import execute_batch, execute_query, execute_write
from ingestion.fetcher import FetchedArticle
from configurables.config import logger 

def load_exisiting_dicts() -> list[dict]:
    rows = execute_query("SELECT url, content_hash FROM articles")
    return { row["url"]: row["content_hash"] for row in rows }

def insert_articles(article: FetchedArticle, content_hash: str) -> str | None:
    "insert new article with pending status."

    sql = """
        INSERT INTO articles (url, content_hash, published_at, source, title, author, ingest_status)
        VALUES (%s, %s, %s, %s, %s, %s, 'pending')
        ON CONFLIFT (url) DO NOTHING
        RETURNING id    
    """

    params = (
        article.url,
        content_hash,
        article.published_at,
        article.src,
        article.title,
        article.author,
    )

    try:
        rows = execute_query(sql=sql, params=params)
        if not rows:
            logger.info("article already exists, skipping inserting for: %s", article.url)
            return None
        return str(rows[0]["id"])

    except Exception as e:
        logger.error("Failed inserting error: %s, for %s", e, article.url)
        return None

def update_article_status(article_id: str, status: str) -> None:
    try:
        sql = "UPDATE articles SET ingest_status = %s WHERE id = %s"
        params = (status, article_id)

        execute_write(sql=sql, params=params)

    except Exception as e:
        logger.error("Failed updating status error: %s, skipping article: %s", article_id)

def insert_nodes(nodes: list[dict]) -> None:
    "batch insert all the chunks for the same article"

    try: 
        sql = """
            INSERT INTO article_nodes (article_id, embedding, node_type, chunk_index, chunk_text, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
        """

        params_list = [(
            node["article_id"],
            node["embedding"],         
            node["node_type"],
            node["chunk_index"],
            node["chunk_text"],
            json.dumps(node["metadata"]), # psycopg2 expects a JSON string not python dict, then it serialises the JSON string to JSONB for postgres 

        ) for node in nodes]

        execute_batch(sql=sql, params_list=params_list)

    except Exception as e:
        logger.error("failed inserting node error: %s, skipping node: %s", e, nodes[0]["article_id"])
    
