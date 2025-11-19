import os
import hashlib
import torch
import chromadb
import networkx as nx
import pandas as pd
#from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------- CONFIG ----------------
CONV_FILE = "rag_documents.txt"
CHROMA_PATH = "./rag_db"
GRAPH_FILE = "graph_rag.gexf"

# fast-mode parameters
MAX_TOKENS = 1000
CHUNK_SIZE = 1500
BATCH_SIZE = 20
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # small, GPU optimized
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---------------- UTILITIES ----------------
def file_hash(path):
    """Create a unique hash of the file to avoid re-ingesting."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def chunk_text(text, chunk_size=CHUNK_SIZE):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# ---------------- Conversation Memory ----------------
def load_documents():
    if not os.path.exists(CONV_FILE):
        return []
    with open(CONV_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def save_response_to_docs(response):
    with open(CONV_FILE, "a", encoding="utf-8") as f:
        f.write(response + "\n")

# ---------------- Embedding Function ----------------
class MiniLMEmbeddings(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)

    def __call__(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

embedding_function = MiniLMEmbeddings()

# ---------------- ChromaDB ----------------
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="rag_collection", embedding_function=embedding_function)

# ---------------- GraphRAG ----------------
graph = nx.DiGraph()
if os.path.exists(GRAPH_FILE):
    graph = nx.read_gexf(GRAPH_FILE)

def save_graph():
    nx.write_gexf(graph, GRAPH_FILE)

def add_to_graph_from_text(text):
    """Adds simplified semantic structure into GraphRAG."""
    lines = [l.strip() for l in text.split(".") if l.strip()]
    prev_node = None
    for line in lines:
        node_id = f"n{len(graph)}"
        graph.add_node(node_id, text=line[:200])
        if prev_node:
            graph.add_edge(prev_node, node_id, relation="follows")
        prev_node = node_id
    save_graph()

# ---------------- Model ----------------
print("🔧 Loading model on GPU ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to(device)

llm = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# ---------------- RAG Query ----------------
def rag_query(question, k=3, store_response=True):
    results = collection.query(query_texts=[question], n_results=k)
    context = "\n".join(results["documents"][0]) if results["documents"] else ""

    prompt = f"""
You are a helpful assistant with access to uploaded documents.
Use the context below, which comes from one or more PDF or CSV files, 
to answer the question accurately and concisely. 
If the answer isn't in the context, say you don't see it in the document.

Context:
{context}

Question: {question}
Answer:
"""
    answer = llm(prompt, max_new_tokens=MAX_TOKENS, temperature=0.7, do_sample=False)[0]["generated_text"].strip()

    if store_response:
        condensed_prompt = f"Summarize briefly for memory storage: {answer}"
        summary = llm(condensed_prompt, max_new_tokens=40, temperature=0.5, do_sample=False)[0]["generated_text"].strip()
        save_response_to_docs(summary)
        add_to_graph_from_text(summary)

    return answer

# ---------------- Document Feeding ----------------
def feed_document_to_memory(file_path):
    """
    Reads PDF/CSV/TXT files, extracts text safely (ignoring XML/metadata),
    chunks, batches, stores in Chroma, and updates GraphRAG.
    """
    import traceback
    import os

    ext = os.path.splitext(file_path)[1].lower()
    file_id = file_hash(file_path)

    # Avoid re-ingesting same file
    if os.path.exists("ingested_files.txt"):
        with open("ingested_files.txt", "r") as f:
            if file_id in f.read():
                print(f"⚠️ {file_path} already ingested.")
                return "⚠️ Already ingested."

    print(f"📘 Ingesting {file_path} ...")
    text = ""

    # ---------------- PDF Handling ----------------
    if ext == ".pdf":
        try:
            # Safe extraction: only page content, ignores XML metadata
            from pdfminer.high_level import extract_text
            text = extract_text(file_path)
        except Exception as e:
            print(f"⚠️ pdfminer extraction failed: {e}")
            traceback.print_exc()
            text = ""

        # Fallback OCR if pdfminer yields nothing
        if not text.strip():
            try:
                from pdf2image import convert_from_path
                import pytesseract
                print("🧠 Using OCR fallback for PDF...")
                pages = convert_from_path(file_path)
                text = "\n".join(pytesseract.image_to_string(p) for p in pages)
            except Exception as e:
                print(f"❌ OCR failed: {e}")
                return f"❌ Could not extract text from {file_path}"

    # ---------------- CSV Handling ----------------
    elif ext == ".csv":
        try:
            import pandas as pd
            df = pd.read_csv(file_path)
            text = "\n".join(df.astype(str).apply(lambda x: ", ".join(x), axis=1).tolist())
        except Exception as e:
            print(f"❌ CSV read failed: {e}")
            return f"❌ Could not read {file_path}"

    # ---------------- TXT/MD Handling ----------------
    elif ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            print(f"❌ Text read failed: {e}")
            return f"❌ Could not read {file_path}"

    else:
        print(f"⚠️ Unsupported file type: {ext}")
        return f"⚠️ Unsupported file type: {ext}"

    if not text.strip():
        print(f"⚠️ No readable text found in {file_path}")
        return f"⚠️ No readable text found in {file_path}"

    # ---------------- Chunk & Batch ----------------
    chunks = chunk_text(text)
    ids, docs = [], []

    for i, chunk in enumerate(chunks):
        ids.append(f"{file_id}_{i}")
        docs.append(chunk)

        # Batch add every BATCH_SIZE or at end
        if len(docs) >= BATCH_SIZE or i == len(chunks) - 1:
            collection.add(documents=docs, ids=ids)
            add_to_graph_from_text(" ".join(docs[:3]))  # small summary in graph
            docs, ids = [], []

    # Mark file as ingested
    with open("ingested_files.txt", "a") as f:
        f.write(file_id + "\n")

    print(f"✅ Finished ingesting {file_path}")
    return f"✅ Finished ingesting {file_path}"
