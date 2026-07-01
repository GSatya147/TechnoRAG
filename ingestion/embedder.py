import voyageai

from configurables.config import VOYAGE_API_KEY, BATCH_SIZE, VOYAGE_EMBEDDING_MODEL, logger

class Embedder:
    def __init__(self):

        try:
            self.vo_client = voyageai.Client(api_key=VOYAGE_API_KEY)
        
        except Exception as e:
            logger.error(e)

    def article_embedder(self, chunks_dict: list[dict]):
        try:
            embeddings = []

            running_tokens = 0
            current_batch = []
            for chunk in chunks_dict:
                if chunk["tokens_count"] + running_tokens >= BATCH_SIZE:
                    embeddings+=self.vo_client.embed(texts=current_batch, model=VOYAGE_EMBEDDING_MODEL, output_dimension=1024).embeddings
                    running_tokens = 0
                    current_batch = []
                
                running_tokens+=chunk["tokens_count"]
                current_batch.append(chunk["chunk_text"])
            
            if current_batch:
                embeddings+=self.vo_client.embed(texts=current_batch, model=VOYAGE_EMBEDDING_MODEL).embeddings

            return embeddings
        
        except Exception as e:
            logger.error(e)

    def embed_and_modify(self, chunks_dict: list[dict]):
        embeddings = self.article_embedder(chunks_dict=chunks_dict)

        if embeddings:
            for chunk, embedding in zip(chunks_dict, embeddings):
                chunk["embedding"] = embedding

            return chunks_dict
        
        return None

