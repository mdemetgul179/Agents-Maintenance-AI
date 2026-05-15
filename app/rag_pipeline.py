import chromadb

from sentence_transformers import SentenceTransformer

from llm_engine import analyze_ticket

# Load embedding model
model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

# Load ChromaDB
client = chromadb.PersistentClient(
    path="../chroma_db"
)

collection = client.get_collection(
    name="maintenance_tickets"
)

# User query
query = input("Enter maintenance issue: ")

# Create embedding
query_embedding = model.encode(query).tolist()

# Search similar tickets
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# Retrieved documents
retrieved_docs = results["documents"][0]

# Print retrieved tickets
print("\nTop Similar Tickets:\n")

for i, doc in enumerate(retrieved_docs):

    print(f"Result {i+1}")
    print("-" * 50)
    print(doc)
    print()

# LLM analysis
analysis = analyze_ticket(
    query,
    retrieved_docs
)

# Print AI response
print("\nAI Maintenance Analysis:\n")
print(analysis)