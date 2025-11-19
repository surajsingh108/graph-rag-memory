import chromadb
import os
import sentence_transformers
import shutil

def clear_model_cache():
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print("Model cache cleared.")
    else:
        print("Model cache directory not found.")


def clear_chromadb_cache():
    """Clears the ChromaDB collection."""
    try:
        client = chromadb.PersistentClient(path="./rag_db")
        client.delete_collection("rag_collection")
        print("ChromaDB collection 'rag_collection' cleared.")
    except Exception as e:
        print(f"Error clearing ChromaDB: {e}")

def clear_sentence_transformer_cache():
    """Clears the Sentence Transformers model cache."""
    try:
        # Delete the local model files
        model_dir = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")._model_path
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            print(f"Sentence Transformers model cache cleared at: {model_dir}")
    except Exception as e:
        print(f"Error clearing Sentence Transformers cache: {e}")

def clear_huggingface_cache():
    """Clears the Hugging Face model cache."""
    try:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Hugging Face model cache cleared at: {cache_dir}")
    except Exception as e:
        print(f"Error clearing Hugging Face cache: {e}")

if __name__ == "__main__":
    print("Clearing caches...")
    clear_chromadb_cache()
    clear_sentence_transformer_cache()
    clear_huggingface_cache()
    clear_model_cache()
    print("Cache clearing complete.")

