```python
import chromadb
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from chromadb.api.types import EmbeddingFunction

```

    c:\Users\suraj\LLM_projects\RAG_memory\LLM_venv\lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    


```python
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

```


```python

documents = load_documents()
ids = [str(i) for i in range(len(documents))]

```


```python
# -------- Embedding Function Wrapper --------
class MiniLMEmbeddings(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, texts):
        return self.model.encode(texts).tolist()

    def name(self):
        return "all-MiniLM-L6-v2"


embedding_function = MiniLMEmbeddings()
```


```python
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

```


```python
# -------- Load an Open (Non-Gated) Model --------
# ✅ Replace Gemma due to access restrictions
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)
llm2 = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

```

    `torch_dtype` is deprecated! Use `dtype` instead!
    Device set to use cuda:0
    Device set to use cuda:0
    


```python

# -- function to reduce the size of the RAG document -- 
def enforce_doc_size_limit(max_chars=10000, summary_chars=5000):
    if not os.path.exists(FILE_PATH):
        return

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    if len(content) <= max_chars:
        return  # ✅ Safe — do nothing

    print("⚠️ Document too large — summarizing and replacing content...")

    # Summarize the entire document via LLM
    prompt = f"""
Summarize the following conversation history into no more than {summary_chars} characters 
while preserving key information for future context:

{content}

Summary:
    """

    summary = llm2(prompt, max_new_tokens=summary_chars, do_sample=True)[0]["generated_text"].strip()

    # ✅ Replace full file with summary only
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    # Rebuild ChromaDB fully with new content
    new_docs = load_documents()
    new_ids = [str(i) for i in range(len(new_docs))]

    client.delete_collection("rag_collection")
    new_collection = client.get_or_create_collection(
        name="rag_collection",
        embedding_function=embedding_function
    )
    new_collection.add(documents=new_docs, ids=new_ids)

    print("✅ Replacement summary stored and ChromaDB rebuilt.")

```


```python

# -------- RAG Query Function --------
def rag_query(question, k=3, store_response=True):
    results = collection.query(query_texts=[question], n_results=k)
    context = "\n".join(results["documents"][0])

    prompt = f"""
Answer the question based largely on the context below. Treat the context below as a history of our past conversations. 
The last ones being more recent ones. If the answer is not in the context, say something related to it. 

Context:
{context}

Question: {question}
Answer:
    """

    answer = llm(prompt, max_new_tokens=150, do_sample=True)[0]["generated_text"].strip()

    if store_response:
        prompt2 = f"""
Summarise the text after the colons in a format of a discussion with a question and an answer: {answer}"""

        Store_answer = llm2(prompt2, max_new_tokens=150, do_sample=True)[0]["generated_text"].strip()
        save_response_to_docs(Store_answer)
        enforce_doc_size_limit()
    return answer


```


```python

# -------- Test Query --------
print("\nRAG Response:")
print(rag_query("What have we been discussing?"))

```

    
    RAG Response:
    Answer the question based largely on the context below. Treat the context below as a history of our past conversations. 
    The last ones being more recent ones. If the answer is not in the context, say something related to it. 
    
    Context:
    However, without explicit confirmation, it cannot be definitively stated what exactly "we" were discussing at the time this conversation ended. Therefore, the best response to your question would be that we have been discussing methods for delving deeper into certain topics within different subject areas. For instance, we might be exploring scientific concepts, examining technological advancements, analyzing cultural phenomena, or investigating historical events, among others. The focus is on offering greater depth and detail regarding these subjects by potentially sharing more comprehensive information or illustrative examples upon request. This approach allows for flexibility and adaptability depending on individual interests and requirements. If you have a specific interest in one of these areas, feel free to ask questions or seek clarification. The goal is to facilitate meaningful discussions and enhance
    If you're interested in delving deeper into any particular area mentioned above, I can provide more detailed insights or additional examples. Just let me know your preference!
    Let me know if there's anything else I can help clarify.
    
    Question: What have we been discussing?
    Answer:
         We have been discussing methods for delving deeper into various topics across different subject areas such as science, technology, culture, and history. Our approach involves sharing more comprehensive information or illustrative examples upon request. If you have a specific interest, please let us know so we can tailor our discussion accordingly. Let me know if you need further details or clarifications! Answer:
    
    We have been discussing methods for delving deeper into various topics across different subject areas such as science, technology, culture, and history. Our approach involves sharing more comprehensive information or illustrative examples upon request. If you have a specific interest, please let us know so we can tailor our discussion accordingly. Let me know if you need further details or clarifications!
    
