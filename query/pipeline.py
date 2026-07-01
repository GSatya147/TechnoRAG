from query.classifier import query_classifier
from query.retriever import build_fusion_retriever
from query.reranker import reranker
from query.mmr import MMRPostprocessor, attach_embeddings
from query.temporal import build_nodes_dict, temporal_assessment
from query.generator import build_user_turn, get_final_response
from query.faithfulness import faithfulness_judgement
from query.response import exctract_citations, build_response
from configurables.config import logger

def query_pipeline_run(query):

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
    
    return final_response

if __name__=="__main__":
    query = input(">> ")

    final_response = query_pipeline_run(query=query)
    logger.info("%s", final_response)
