from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str):
        return self.model.encode(text).tolist()
    
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
        global _embedding_service
        if _embedding_service is None:
            _embedding_service = EmbeddingService()
        return _embedding_service