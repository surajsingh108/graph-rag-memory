import os
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import tiktoken

# --- Configuration ---
MODEL_PATH = "C:\Users\suraj\.ollama\models\manifests\registry.ollama.ai\library\gemma3\4b"  # Update this to your Ollama model path
CONTEXT_WINDOW = 2048 # Adjust this according to Gemma3's context window
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # A good balance of speed and accuracy
NUM_EMBEDDINGS = 10 #  Increase for a larger searchable index
TOKENIZER_NAME = "gpt2"  # The tokenizer to use for Gemma3
# --- End Configuration ---



def load_model(model_path):
    """Loads the Gemma3 model using llama-cpp-python."""
    llm = Llama(model_path=model_path, n_ctx=CONTEXT_WINDOW, verbose=False)  # Disable verbose output
    return llm


def load_embedding_model():
    """Loads the sentence-transformers model."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def create_index(model, texts):
    """Creates an embedding index."""
    embeddings = model.encode(texts)
    return embeddings


def query_index(embeddings, query, model):
    """Queries the index for relevant documents."""
    query_embedding = model.encode(query)
    distances, indices = model.cos_sim(query_embedding, embeddings)  # Use cosine similarity
    return distances, indices


def generate_response(llm, query, distances, indices, tokenizer_name):
    """Generates a response using the LLM and retrieved context."""
    tokenizer = tiktoken.get_encoding(tokenizer_name)

    # Create the prompt
    prompt = f"Context: {tokenizer.decode(indices[0])}\n\nQuestion: {query}\n\nAnswer:"

    output = llm(prompt, max_tokens=256, stop=["Q:", "\n"], echo=False)
    return output['choices'][0]['text']


if __name__ == "__main__":
    # Sample Data - Replace with your actual documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "This is another example document about Gemma 3.",
        "RAG is a technique that combines retrieval with LLMs.",
        "Python is a popular programming language."
    ]
    
    # Load components
    llm = load_model(MODEL_PATH)
    embedding_model = load_embedding_model()

    # Create the index
    embeddings = create_index(embedding_model, documents)

    # Get user query
    query = "What is RAG?"

    # Query the index
    distances, indices = query_index(embeddings, query, embedding_model)

    # Generate response
    response = generate_response(llm, query, distances, indices, TOKENIZER_NAME)

    print(f"Query: {query}")
    print(f"Response: {response}")
