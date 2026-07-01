import voyageai

from configurables.config import VOYAGE_API_KEY, VOYAGE_EMBEDDING_MODEL, logger

def query_embedder(user_query: str):
    try:
        vo_client = voyageai.Client(api_key=VOYAGE_API_KEY)

        embeddings = vo_client.embed(texts=user_query, output_dimension=1024, model=VOYAGE_EMBEDDING_MODEL).embeddings

        return embeddings[0]
    except Exception as e:
        logger.error(e)

if __name__=="__main__":
    embeddings = query_embedder("Hello")
    logger.info("%s", embeddings)