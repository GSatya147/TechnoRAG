from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.openai_like import OpenAILike

from db.db_client import execute_query
from query.query_embedder import query_embedder
from configurables.config import logger, DEEPSEEK_API_KEY, DEEPSEEK_MODEL

class DenseRetriever(BaseRetriever):
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        super().__init__()
 
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_embedding = query_embedder(query_bundle.query_str)
 
        sql = """
            SELECT
                id,
                article_id,
                chunk_text,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM article_nodes
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        rows = execute_query(sql, (query_embedding, query_embedding, self.top_k))
 
        return [
            NodeWithScore(
                node=TextNode(
                    text=row["chunk_text"],
                    id_=str(row["id"]),
                    metadata={**row["metadata"], "article_id": str(row["article_id"])},
                ),
                score=row["similarity"],
            )
            for row in rows
        ]
    
class SparseRetriever(BaseRetriever):
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        super().__init__()
 
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        sql = """
            SELECT
                id,
                article_id,
                chunk_text,
                metadata,
                ts_rank(sparse_tsv, plainto_tsquery('english', %s)) AS score
            FROM article_nodes
            WHERE sparse_tsv @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s;
        """
        query_text = query_bundle.query_str
        rows = execute_query(sql, (query_text, query_text, self.top_k))
 
        return [
            NodeWithScore(
                node=TextNode(
                    text=row["chunk_text"],
                    id_=str(row["id"]),
                    metadata={**row["metadata"], "article_id": str(row["article_id"])},
                ),
                score=row["score"],
            )
            for row in rows
        ]
    
def build_fusion_retriever(top_k: int = 10) -> QueryFusionRetriever:
    dense_retriever = DenseRetriever(top_k=top_k)
    sparse_retriever = SparseRetriever(top_k=top_k)

    deepseek_llm = OpenAILike(
        model=DEEPSEEK_MODEL,
        api_base="https://api.deepseek.com/v1",
        api_key=DEEPSEEK_API_KEY,
        is_chat_model=True,
    )

    return QueryFusionRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        mode="reciprocal_rerank",  
        similarity_top_k=top_k,
        num_queries=1,             
        use_async=False,            # flip to True + use aretrieve() once you're on FastAPI's event loop
        verbose=True,
        llm=deepseek_llm
    )

if __name__=="__main__":
   retriever =  build_fusion_retriever()
   results = retriever.retrieve("what is OpenAi?")
   logger.info("%s", results)