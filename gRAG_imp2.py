import os
import chromadb
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import networkx as nx
import pandas as pd
from PyPDF2 import PdfReader

# ------------------ File paths ------------------
CONV_FILE = "rag_documents.txt"
CHROMA_PATH = "./rag_db"
GRAPH_FILE = "graph_rag.gexf"

# ------------------ Load / Save Conversation ------------------
def load_documents(file_path=CONV_FILE):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]
    return docs

def save_response_to_docs(response, file_path=CONV_FILE):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(response + "\n")

# ------------------ Embedding Function ------------------
class MiniLMEmbeddings(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, texts):
        return self.model.encode(texts).tolist()

    def name(self):
        return "all-MiniLM-L6-v2"

embedding_function = MiniLMEmbeddings()

# ------------------ Chroma Setup ------------------
client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    client.delete_collection("rag_collection")
except:
    pass

collection = client.get_or_create_collection(
    name="rag_collection",
    embedding_function=embedding_function
)

# Add documents loaded from file
documents = load_documents()
ids = [str(i) for i in range(len(documents))]
if documents:
    collection.add(documents=documents, ids=ids)

# ------------------ Load LLM ------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# ------------------ GraphRAG ------------------
graph = nx.DiGraph()
if os.path.exists(GRAPH_FILE):
    graph = nx.read_gexf(GRAPH_FILE)

def save_graph():
    nx.write_gexf(graph, GRAPH_FILE)

def add_to_graph_from_text(text):
    """Add nodes/edges from text; very simple placeholder logic."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        node_id = f"line_{len(graph)}"
        graph.add_node(node_id, text=line[:200])
        if i > 0:
            prev_node = f"line_{len(graph)-2}"
            graph.add_edge(prev_node, node_id, relation="follows")
    save_graph()

# ------------------ RAG Query ------------------
def rag_query(question, k=3, store_response=True):
    results = collection.query(query_texts=[question], n_results=k)
    context = "\n".join(results["documents"][0]) if results["documents"] else ""
    prompt = f"""
Answer the question based largely on the context below. Treat the context as a memory of past interactions.
Context:
{context}
Question: {question}
Answer:
    """
    answer = llm(prompt, max_new_tokens=150, do_sample=True)[0]["generated_text"].strip()

    if store_response:
        # Store condensed version
        prompt2 = f"Summarise to keywords only for memory. You have schizophrenia: {answer}"
        Store_answer = llm(prompt2, max_new_tokens=50, do_sample=True)[0]["generated_text"].strip()
        save_response_to_docs(Store_answer)
        add_to_graph_from_text(Store_answer)

    return answer

# ------------------ Document Feeding ------------------
def chunk_text(text, chunk_size=2000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def feed_document_to_memory(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        text = "\n".join(df.astype(str).apply(lambda x: ", ".join(x), axis=1).tolist())
    elif ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Chunk text to avoid memory issues
    chunks = chunk_text(text, chunk_size=2000)
    for chunk in chunks:
        # Add to conversation memory
        snippet = chunk[:2000]
        save_response_to_docs(f"[DOC-UPLOAD] {snippet}")
        # Add to Chroma
        new_id = str(len(collection.get()["ids"]))
        collection.add(documents=[chunk], ids=[new_id])
        # Add to GraphRAG
        add_to_graph_from_text(chunk)
    print(f"✅ Ingested document: {file_path}")
