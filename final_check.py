from grag import RAG, Config
from grag.ingest import ingest
from grag.extractor import extract_triples

rag = RAG(Config())
print("RAG created, memory empty:", rag.memory.is_empty())
print("Graph stats:", rag.graph.stats())

class MockLLM:
    def extract_triples(self, text):
        return '[{"subject": "Einstein", "relation": "born_in", "object": "Ulm"}]'

triples = extract_triples("Albert Einstein was born in Ulm.", MockLLM())
print("Extracted triples:", triples)
print("All imports and core logic OK.")
