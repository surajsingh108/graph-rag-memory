import os
import time
import json
import math
import networkx as nx
import chromadb
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ----------------- Config -----------------
CONV_FILE = "rag_conversations.jsonl"   # stores conversation entries (json lines)
GRAPH_FILE = "graph_rag.gexf"           # networkx graph store
CHROMA_PATH = "./rag_db"
CHROMA_COLLECTION = "rag_collection"

# Limits & behavior
CONV_CHAR_LIMIT = 3000   # when conversation file exceeds this (chars), summarize oldest 25%
GRAPH_NODE_LIMIT = 80    # when graph nodes exceed this, summarize oldest 25%
OLD_QUARTER = 0.25       # fraction to summarize (oldest 25%)

# LLM / model
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # adjust if you use another model
MAX_ANSWER_TOKENS = 200

# ----------------- Utility: Time -----------------
def now_ts():
    return int(time.time())

# ----------------- Conversation store -----------------
def append_conversation(role: str, text: str):
    entry = {"ts": now_ts(), "role": role, "text": text}
    with open(CONV_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def load_conversations():
    if not os.path.exists(CONV_FILE):
        return []
    out = []
    with open(CONV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # skip malformed line
                continue
    # sort by timestamp ascending (oldest first)
    out.sort(key=lambda x: x.get("ts", 0))
    return out

def write_conversations(entries):
    # overwrite conversation file with list of entries (each a dict)
    with open(CONV_FILE, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

# ----------------- Embedding wrapper -----------------
class MiniLMEmbeddings(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def __call__(self, texts):
        return self.model.encode(texts).tolist()
    def name(self):
        return "all-MiniLM-L6-v2"

# ----------------- Chroma setup -----------------
embedding_function = MiniLMEmbeddings()
client = chromadb.PersistentClient(path=CHROMA_PATH)

# remove collection if exists, then create / get
try:
    client.delete_collection(CHROMA_COLLECTION)
except Exception:
    pass

collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    embedding_function=embedding_function
)

def rebuild_chroma_from_conversations():
    """
    We store only assistant messages and conversation summaries (assistant role or summaries)
    as documents in Chroma for retrieval. Each document id is a stable integer string.
    """
    convs = load_conversations()
    docs = []
    ids = []
    for i, e in enumerate(convs):
        # store assistant text or explicit 'summary' items (we treat any role=='assistant' as doc)
        if e.get("role") == "assistant":
            docs.append(e.get("text", ""))
            ids.append(str(i))
        # Optionally store summaries even if role isn't assistant: if we want keep both, uncomment
        # elif e.get("role") == "summary":
        #     docs.append(e.get("text",""))
        #     ids.append(str(i))
    # Remove existing collection and recreate
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    col = client.get_or_create_collection(name=CHROMA_COLLECTION, embedding_function=embedding_function)
    if docs:
        col.add(documents=docs, ids=ids)
    return col

# initial build
collection = rebuild_chroma_from_conversations()

# ----------------- LLM pipeline (single shared) -----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

# ----------------- GraphRAG (NetworkX) -----------------
graph = nx.DiGraph()
if os.path.exists(GRAPH_FILE):
    try:
        graph = nx.read_gexf(GRAPH_FILE)
    except Exception:
        graph = nx.DiGraph()

def save_graph():
    nx.write_gexf(graph, GRAPH_FILE)

def add_to_graph_from_text(text, ts=None):
    """
    Use LLM to extract triples from text and add to the graph.
    We expect LLM to output JSON list of triples: [{"entity1":"A","relation":"r","entity2":"B"}, ...]
    Each created node will have attribute 'created_ts' for age-based pruning.
    """
    ts = ts or now_ts()
    prompt = f"""
Extract the key entities and their relationships from the text below.
Return output as a JSON list of triples with keys: entity1, relation, entity2.
Do not include additional commentary.

Text:
{text}

Output:
"""
    try:
        raw = llm(prompt, max_new_tokens=300, do_sample=False)[0]["generated_text"].strip()
    except Exception as e:
        print("LLM triple extraction failed:", e)
        return

    # Try to parse JSON; LLM might return trailing text; extract first JSON-looking substring
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        # attempt to find first '[' ... ']' block
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end+1])
            except Exception:
                parsed = None
    if not parsed or not isinstance(parsed, list):
        # fallback: create a single generic node
        node_name = f"concept_{ts}"
        graph.add_node(node_name, text=text[:200], created_ts=ts)
        save_graph()
        return

    for t in parsed:
        e1 = t.get("entity1") or t.get("subject") or None
        rel = t.get("relation") or t.get("predicate") or "related_to"
        e2 = t.get("entity2") or t.get("object") or None
        if not e1 or not e2:
            continue
        # add nodes (with created_ts if new)
        if not graph.has_node(e1):
            graph.add_node(e1, created_ts=ts)
        if not graph.has_node(e2):
            graph.add_node(e2, created_ts=ts)
        # add edge with relation and timestamp
        graph.add_edge(e1, e2, relation=rel, created_ts=ts)
    save_graph()

# ----------------- Graph summarization helper -----------------
def summarize_graph_subgraph(nodes_to_summarize):
    """
    Given a list of nodes (oldest quarter), create a textual description of their subgraph
    and ask LLM to summarize into ~300 characters. Return the summary string.
    """
    # build subgraph textual representation
    edges = []
    for a, b, d in graph.subgraph(nodes_to_summarize).edges(data=True):
        rel = d.get("relation", "related_to")
        edges.append(f"{a} --{rel}--> {b}")
    # include node list
    node_list = ", ".join(nodes_to_summarize[:50])
    raw_text = "Nodes: " + node_list + "\nEdges:\n" + ("\n".join(edges[:200]) if edges else "none")
    prompt = f"""
You are given a set of nodes and relations from a knowledge graph. Produce a concise summary (not more than 300 characters)
that captures the main themes and facts represented in this subgraph. Output only the summary.

Subgraph:
{raw_text}

Summary:
"""
    summary = llm(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"].strip()
    return summary

# ----------------- Conversation summarization helper -----------------
def summarize_text_entries(entries, char_limit=500):
    """
    entries: list of conversation dicts (oldest first)
    Return a concise summary string of length <= char_limit characters.
    """
    concat = "\n".join([f"{e['role']}: {e['text']}" for e in entries])
    prompt = f"""
Summarize the following conversation excerpts into a concise summary no longer than {char_limit} characters.
Keep the important facts and discard minor chit-chat. Output only the summary.

Conversation:
{concat}

Summary:
"""
    summary = llm(prompt, max_new_tokens=math.ceil(char_limit / 4), do_sample=True)[0]["generated_text"].strip()
    # ensure truncated to char_limit
    if len(summary) > char_limit:
        summary = summary[:char_limit].rsplit(" ", 1)[0]
    return summary

# ----------------- Partial (oldest quarter) summarizers -----------------
def summarize_oldest_conversations_if_needed():
    convs = load_conversations()
    total_chars = sum(len(e.get("text","")) for e in convs)
    if total_chars <= CONV_CHAR_LIMIT:
        return False
    n = len(convs)
    if n == 0:
        return False
    q = max(1, int(math.ceil(n * OLD_QUARTER)))
    oldest = convs[:q]        # oldest quarter (preserve order)
    remainder = convs[q:]     # keep recent 75% untouched

    # Summarize oldest quarter into a single summary entry with role 'summary' or 'assistant'
    summary_text = summarize_text_entries(oldest, char_limit=500)
    summary_entry = {"ts": now_ts(), "role": "assistant", "text": f"[SUMMARY-ARCHIVE] {summary_text}"}

    # write new conversations (remainder + summary at the end)
    new_convs = remainder + [summary_entry]
    write_conversations(new_convs)

    # Add the summary to Chroma as an assistant document and rebuild chroma
    rebuild_chroma_from_conversations()

    # Add the summary to graph as a condensed chunk
    add_to_graph_from_text(summary_text, ts=summary_entry["ts"])
    print(f"✅ Summarized oldest {q} conversation entries into one summary entry.")
    return True

def summarize_oldest_graph_if_needed():
    node_count = graph.number_of_nodes()
    if node_count <= GRAPH_NODE_LIMIT:
        return False
    # pick oldest quarter of nodes by created_ts (nodes may be missing created_ts if added manually)
    nodes_with_ts = [(n, graph.nodes[n].get("created_ts", 0)) for n in graph.nodes()]
    nodes_with_ts.sort(key=lambda x: x[1])  # oldest first
    q = max(1, int(math.ceil(len(nodes_with_ts) * OLD_QUARTER)))
    oldest_nodes = [n for n, ts in nodes_with_ts[:q]]

    # produce summary of this subgraph
    summary = summarize_graph_subgraph(oldest_nodes)
    # remove oldest nodes and their edges
    # collect neighbors to reconnect summary node
    neighbors = set()
    for n in oldest_nodes:
        # neighbors previously connected (in or out)
        for nbr in graph.predecessors(n):
            if nbr not in oldest_nodes:
                neighbors.add(nbr)
        for nbr in graph.successors(n):
            if nbr not in oldest_nodes:
                neighbors.add(nbr)
    # remove nodes
    graph.remove_nodes_from(oldest_nodes)

    # add a summary node and connect to neighbors
    summary_node = f"summary_node_{now_ts()}"
    graph.add_node(summary_node, text=summary[:400], created_ts=now_ts(), is_summary=True)
    for nbr in neighbors:
        # connect summary_node to neighbor (summary_of -> neighbor)
        graph.add_edge(summary_node, nbr, relation="summary_of", created_ts=now_ts())
    save_graph()

    print(f"✅ Summarized and removed {len(oldest_nodes)} graph nodes; created {summary_node}.")
    return True

# ----------------- Build graph context string for prompt -----------------
def build_graph_context(max_edges=20):
    edges = list(graph.edges(data=True))[:max_edges]
    lines = []
    for a, b, d in edges:
        rel = d.get("relation", "related_to")
        lines.append(f"{a} --{rel}--> {b}")
    return "\n".join(lines)

import pandas as pd
from PyPDF2 import PdfReader
import os

def feed_document_to_memory(file_path):
    """
    Reads a PDF, CSV, or TXT file and ingests its content into your RAG system.
    """
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
    
    # Store a small snippet in conversation file (optional)
    snippet = text[:2000]
    save_response_to_docs(f"[DOC-UPLOAD] {snippet}")
    
    # Add to Chroma collection
    new_id = str(len(collection.get()["ids"]))
    collection.add(documents=[text], ids=[new_id])
    
    # Optionally update GraphRAG
    # If you have a function like add_to_graph_from_text(), call it here:
    if "add_to_graph_from_text" in globals():
        add_to_graph_from_text(text)
    
    print(f"✅ Ingested document: {file_path}")


# ----------------- RAG query function -----------------
def rag_query(question, k=3, store=True):
    """
    1. Append user question to conversation file
    2. Query chroma for context (assistant docs)
    3. Build prompt using chroma docs + graph context
    4. Generate answer-only output
    5. Append assistant answer to convo file, add to graph and chroma
    6. Run partial summarizers if needed (convo & graph)
    """
    # append user query
    append_conversation("user", question)

    # Query chroma for context documents
    try:
        results = collection.query(query_texts=[question], n_results=k)
        docs = results.get("documents", [[]])[0]
        context_docs = "\n".join(docs) if docs else ""
    except Exception:
        context_docs = ""

    graph_context = build_graph_context()

    prompt = f"""
Answer the question concisely and directly using the context. Output only the final answer (no repetition of the question).
Context documents:
{context_docs}

Graph knowledge:
{graph_context}

Question:
{question}

Answer:
"""
    # Generate answer-only
    raw = llm(prompt, max_new_tokens=MAX_ANSWER_TOKENS, do_sample=True)[0]["generated_text"].strip()
    # best-effort strip: remove prompt if model echoed it (rare with "output only the final answer")
    answer = raw
    if raw.startswith(prompt):
        answer = raw.replace(prompt, "").strip()

    # append assistant answer to conversation file
    append_conversation("assistant", answer)

    # add assistant answer doc to chroma (rebuild)
    rebuild_chroma_from_conversations()

    # add to graph (extract triples)
    add_to_graph_from_text(answer)

    # run partial summarizers (they themselves will rebuild chroma/graph as needed)
    summarize_oldest_conversations_if_needed()
    summarize_oldest_graph_if_needed()

    return answer

# ----------------- Quick helpers (manual) -----------------
def interpret_graph_simple(limit=30):
    """
    Print simple human-readable interpretation of the graph: top relations and most connected nodes.
    """
    print(f"\nGraph: nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}\n")
    for i, (a, b, d) in enumerate(graph.edges(data=True)):
        if i >= limit:
            break
        print(f"{a} --{d.get('relation','related_to')}--> {b}")
    # top nodes
    deg_sorted = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    print("\nTop connected nodes:")
    for n, deg in deg_sorted[:10]:
        print(f"- {n} ({deg})")

# ----------------- Example usage -----------------
if __name__ == "__main__":
    # Example interactive usage:
    print("Starting RAG system. Type a question or 'quit'.\n")
    try:
        while True:
            q = input("You: ").strip()
            if q.lower() in ("quit", "exit"):
                break
            if not q:
                continue
            ans = rag_query(q)
            print("\nAssistant:", ans, "\n")
    finally:
        # Save graph on exit
        save_graph()
