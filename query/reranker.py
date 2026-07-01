from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank

from configurables.config import VOYAGE_API_KEY, logger
from query.retriever import build_fusion_retriever

def reranker(query_str, fused_nodes: list[NodeWithScore]):
    reranker = VoyageAIRerank(
    model="rerank-2.5",
    api_key=VOYAGE_API_KEY,
    top_n=5,
    truncation=True,
    )

    reranked_nodes = reranker.postprocess_nodes(
        fused_nodes,          # list[NodeWithScore] from fusion
        query_str=query_str,  # raw query text, not embedded
    )

    return reranked_nodes

if __name__=="__main__":
    retriever =  build_fusion_retriever()
    fused_nodes = retriever.retrieve("what is OpenAi?")

    reranker_results = reranker(query_str="what is OpenAi?", fused_nodes=fused_nodes)
    logger.info("%s", reranker_results)