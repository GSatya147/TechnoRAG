import regex as re

from query.classifier import query_classifier
from query.retriever import build_fusion_retriever
from query.reranker import reranker
from query.mmr import MMRPostprocessor, attach_embeddings
from query.temporal import build_nodes_dict, temporal_assessment
from query.generator import build_user_turn, get_final_response
from query.faithfulness import faithfulness_judgement
from configurables.config import logger, CITATION_PATTERN
# from client.client import Client

def exctract_citations(answer_text: str, citation_map: dict) -> list[dict]:
    matches: list = re.findall(CITATION_PATTERN, answer_text)
    cited_numbers: set = sorted(set(int(n) for n in matches))

    citations: list[dict] = []
    for n in cited_numbers:
        node = citation_map.get(n)
        if node is None:
            continue # model created new number

        citations.append({
            "n"             : n,
            "url"           : node.metadata.get("url"),
            "title"         : node.metadata.get("title"),
            "published_at"  : node.metadata.get("published_at")
        })
    
    return citations

def build_response(answer_text, citations, judge_result) -> dict:
    return {
        "answer_text"   : answer_text,
        "citations"     : citations,
        "judge_verdict" : judge_result.get("verdict"),
        "judge_claims"  : judge_result.get("flagged_claims", [])
    }

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

    source_chunks, citation_map = build_user_turn(updated_nodes_dict)

    response = get_final_response(user_query=query, query_classification=classification, source_chunks=source_chunks)

    judgment = faithfulness_judgement(response=response, source_chunks=source_chunks)

    citations = exctract_citations(answer_text=response, citation_map=citation_map)
    final_response = build_response(answer_text=response, citations=citations, judge_result=judgment)
    logger.info("%s", final_response)