from datetime import datetime, timezone

import numpy as np
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
 
from db.db_client import execute_query
from query.retriever import build_fusion_retriever
from query.reranker import reranker

class MMRPostprocessor(BaseNodePostprocessor):
    lambda_mult: float = 0.5
    top_n: int = 5
 
    def _postprocess_nodes(self, nodes: list[NodeWithScore], query_bundle: QueryBundle) -> list[NodeWithScore]:
        embeddings = [n.node.embedding for n in nodes]
        if any(e is None for e in embeddings):
            raise ValueError("MMR requires node.embedding set on every candidate")
 
        selected: list[int] = []
        candidates = list(range(len(nodes)))
 
        while candidates and len(selected) < self.top_n:
            if not selected:
                best = max(candidates, key=lambda i: nodes[i].score)
            else:
                def mmr_score(i: int) -> float:
                    relevance = nodes[i].score
                    max_sim = max(self._cosine_sim(embeddings[i], embeddings[j]) for j in selected)
                    return self.lambda_mult * relevance - (1 - self.lambda_mult) * max_sim
 
                best = max(candidates, key=mmr_score)
 
            selected.append(best)
            candidates.remove(best)
 
        return [nodes[i] for i in selected]
 
    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
 
 
def attach_embeddings(nodes: list[NodeWithScore]) -> None:
    node_ids = [n.node.node_id for n in nodes]
    sql = "SELECT id, embedding FROM article_nodes WHERE id = ANY(%s::uuid[]);"
    rows = execute_query(sql, (node_ids,))
    embedding_by_id = {str(row["id"]): row["embedding"] for row in rows}
 
    for n in nodes:
        n.node.embedding = embedding_by_id[n.node.node_id]
 
 
if __name__ == "__main__":
    retriever =  build_fusion_retriever()
    fused_nodes = retriever.retrieve("what is OpenAi?")

    reranked_nodes = reranker(query_str="what is OpenAi?", fused_nodes=fused_nodes)
 
    attach_embeddings(reranked_nodes)
 
    mmr = MMRPostprocessor(lambda_mult=0.5, top_n=5)
    final_nodes = mmr.postprocess_nodes(reranked_nodes)
 
    now = datetime.now(timezone.utc)
    for n in final_nodes:
        published_at = datetime.fromisoformat(n.node.metadata["published_at"])
        print(f"{n.score:.4f} | {n.node.text[:80]}")
        print(f"age days: {(now - published_at).days}")