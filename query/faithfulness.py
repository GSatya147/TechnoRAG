import json

from query.classifier import query_classifier
from query.retriever import build_fusion_retriever
from query.reranker import reranker
from query.mmr import MMRPostprocessor, attach_embeddings
from query.temporal import build_nodes_dict, temporal_assessment
from query.generator import build_user_turn, get_final_response
from configurables.config import logger
from client.client import Client

SYSTEM_PROMPT = """
You are the faithfulness judge for TechnoRAG. You will be given a generated answer and
the numbered source chunks it was allowed to use. Your job is to verify the answer is
fully grounded in those sources — nothing more, nothing less.

CHECK FOR THREE FAILURE MODES
1. unsupported: a claim in the answer that is not actually stated in any cited source,
   or cites a source number that doesn't support the specific claim attached to it.
2. stale_presented_as_current: a claim drawn from a source tagged STALE, but stated in
   the answer without any freshness caveat — presented as if it were current fact.
3. contradicts_source: a claim that actively conflicts with what a cited source says.

RULES
- Judge only against the provided sources. Do not use outside knowledge to judge
  correctness — you are checking grounding, not real-world accuracy.
- A citation is valid only if the specific claim next to it is actually supported by
  that specific chunk. A correct fact with the wrong citation number still counts as
  unsupported.
- Minor rephrasing/summarizing is fine. Flag only claims that add unsupported specifics
  (numbers, dates, names) or unsupported certainty.
- If a STALE source is used WITH a visible caveat ("as of [date]..."), that is CORRECT
  behavior — do not flag it.
- verdict is "fail" if ANY claim is flagged. Otherwise "pass".

EXAMPLES (fixed reference cases)

Example 1 — stale presented as current (the primary failure mode this judge exists for):
Source [2] (STALE, 120 days old): "Claude Opus 4.5 remains Anthropic's top model."
Answer text: "Anthropic's current flagship model is Claude Opus 4.5 [2]."
→ FAIL. issue: stale_presented_as_current. The answer states this as current fact with
no caveat, even though the source is STALE.

Example 2 — stale used correctly, should PASS:
Source [2] (STALE, 120 days old): "Claude Opus 4.5 remains Anthropic's top model."
Answer text: "As of roughly 120 days ago, Claude Opus 4.5 was reported as Anthropic's
top model [2]."
→ PASS. The caveat is present; this is exactly correct use of a stale source.

Example 3 — unsupported specific added:
Source [1]: "OpenAI released GPT-5."
Answer text: "OpenAI released GPT-5 in March 2026 [1]."
→ FAIL. issue: unsupported. The source does not state a release date; the month/year
was not in the source and must be flagged even though it sounds plausible.

OUTPUT
Return STRICT JSON only, matching this schema exactly, no prose outside it:
{
  "verdict": "pass" | "fail",
  "flagged_claims": [
    {"claim": "...", "citation": <int>, "issue": "unsupported" | "stale_presented_as_current" | "contradicts_source", "explanation": "..."}
  ]
}
If verdict is "pass", flagged_claims must be an empty list.
"""

client = Client()

def faithfulness_judgement(response: str, source_chunks: str):
    USER_TURN = f"""
    Generated answer:
    {response}

    Source chunks used:
    {source_chunks}

    Evaluate this answer against these sources per the rules above.
    """

    messages_list: list[dict] = [{
        "role"      : "system",
        "content"   : SYSTEM_PROMPT
    },{
        "role"      : "user",
        "content"   : USER_TURN
    }] 

    try:
        response = client.generate_response(messages_list=messages_list)
        return json.loads(response)
    except Exception as e:
        logger.error("error: %s, response: %s", response)

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
    logger.info("%s", response)

    logger.info("%s", source_chunks)

    judgment = faithfulness_judgement(response=response, source_chunks=source_chunks)
    logger.info("%s", judgment)