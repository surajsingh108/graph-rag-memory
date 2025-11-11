import chromadb
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from chromadb.api.types import EmbeddingFunction


# -------- Read docs from text file --------
FILE_PATH = "rag_documents.txt"

def load_documents(file_path=FILE_PATH):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]
    return docs


def save_response_to_docs(response, file_path=FILE_PATH):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(response + "\n")


documents = load_documents()
ids = [str(i) for i in range(len(documents))]


# -------- Embedding Function Wrapper --------
class MiniLMEmbeddings(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, texts):
        return self.model.encode(texts).tolist()

    def name(self):
        return "all-MiniLM-L6-v2"


embedding_function = MiniLMEmbeddings()


# -------- Chroma Setup --------
client = chromadb.PersistentClient(path="./rag_db")

# Reset for clean run
try:
    client.delete_collection("rag_collection")
except:
    pass

collection = client.get_or_create_collection(
    name="rag_collection",
    embedding_function=embedding_function
)

# Add docs read from file
if documents:
    collection.add(
        documents=documents,
        ids=ids
    )


# -------- Load an Open (Non-Gated) Model --------
# ✅ Replace Gemma due to access restrictions
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)


# -------- RAG Query Function --------
def rag_query(question, k=3, store_response=True):
    results = collection.query(query_texts=[question], n_results=k)
    context = "\n".join(results["documents"][0])

    prompt = f"""
Answer the question based on ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:
    """

    answer = llm(prompt, max_new_tokens=150, do_sample=True)[0]["generated_text"].strip()

    if store_response:
        save_response_to_docs(answer)

    return answer


# -------- Test Query --------
print("\nRAG Response:")
print(rag_query("Who developed the theory of relativity?"))
