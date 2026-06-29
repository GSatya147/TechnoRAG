
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.schema import Document
from sentence_transformers import SentenceTransformer

from configurables.config import logger, CHUNK_OVERLAP, CHUNK_SIZE, VOYAGE_EMBEDDING_MODEL
from ingestion.fetcher import FetchedArticle

splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def _token_counter(text: str):
    try:
        model = SentenceTransformer(VOYAGE_EMBEDDING_MODEL)
        tokenizer = model.tokenizer

        results = tokenizer.encode(text)
        return len(results)

    except Exception as e:
        print(e)

def build_document(article: FetchedArticle) -> None:
    return Document(
        text = article.text,
        metadata = {
            "url":          article.url,
            "title":        article.title,
            "source":       article.src,
            "published_at": article.published_at.isoformat() if article.published_at else None,
            "author":       article.author,
        }
    )

def summary_node(article: FetchedArticle, summary_text: str, article_id: str) -> dict:
    return {
        "article_id":   article_id,
        "node_type":    "summary",
        "chunk_index":  -1,
        "chunk_text":   summary_text,
        "metadata": {
            "url":          article.url,
            "title":        article.title,
            "source":       article.src,
            "published_at": article.published_at.isoformat() if article.published_at else None,
            "author":       article.author,
        },
        "tokens_count":  _token_counter(summary_text),
        "embedding": None, 
    }

def chunk_article(article: FetchedArticle, article_id) -> list[TextNode]:
    document = build_document(article=article)

    nodes: list[TextNode] = splitter.get_nodes_from_documents(documents=[document])

    chunk_dicts: list[dict] = []
    for i, node in enumerate(nodes):
        chunk_dicts.append({
            "article_id":   article_id,
            "node_type":    "chunk",
            "chunk_index":  i,
            "chunk_text":   node.text,
            "metadata":     node.metadata,  # already inherited from Document
            "tokens_count":  _token_counter(node.text),
            "embedding":    None,         
        })
    
    logger.info("Produced %d chunk for %s article", len(chunk_dicts), article.url)
    return chunk_dicts