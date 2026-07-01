from datetime import datetime, timezone

from llama_index.core.schema import NodeWithScore

from query.classifier import query_classifier
from query.retriever import build_fusion_retriever
from query.reranker import reranker
from query.mmr import MMRPostprocessor, attach_embeddings
from configurables.config import logger, TEMPORAL_THRESHOLDS

def calculate_age(node):
    now = datetime.now(timezone.utc)

    if node.metadata["published_at"]:
        published_at = datetime.fromisoformat(node.metadata["published_at"])

        return (now - published_at).days
    return None

def build_nodes_dict(ranked_nodes: list[NodeWithScore]):
    nodes_dict: list[dict] = []

    for node in ranked_nodes:
        nodes_dict.append({
            "node" : node,
            "age" : calculate_age(node=node),
        })

    return nodes_dict

def temporal_assessment(nodes_dict, query_classification):
    intent = query_classification.get("intent")
    threshold = TEMPORAL_THRESHOLDS.get(intent)
    
    for field in nodes_dict:
        if threshold is None:
            field["temporal_assessment"] = "FRESH"  # conceptual - no flag
        
        elif field.get("age") is None:
            field["temporal_assessment"] = "STALE"

        elif field.get("age") > threshold:
            field["temporal_assessment"] = "STALE"
        
        else:
            field["temporal_assessment"] = "FRESH"
    
    return nodes_dict

if __name__=="__main__":
    query = "what is OpenAi?"

    classification = query_classifier(user_query=query)

    retriever =  build_fusion_retriever()
    fused_nodes = retriever.retrieve(query)

    reranked_nodes = reranker(query_str=query, fused_nodes=fused_nodes)
 
    attach_embeddings(reranked_nodes)
 
    mmr = MMRPostprocessor(lambda_mult=0.5, top_n=5)
    post_mmr_nodes = mmr.postprocess_nodes(reranked_nodes)
 
    nodes_dict = build_nodes_dict(ranked_nodes=post_mmr_nodes)
    updated_nodes_dict = temporal_assessment(nodes_dict=nodes_dict, query_classification=classification)

    for node in updated_nodes_dict:
        logger.info("age: %s, assessment: %s\n", node.get("age"), node.get("temporal_assessment"))
