import threading

_model = None
_model_lock = threading.Lock()

def get_sentence_transformer():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                import logging
                logger = logging.getLogger(__name__)
                logger.info("Initializing SentenceTransformer globally...")
                _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model
