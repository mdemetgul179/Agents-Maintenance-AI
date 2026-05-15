import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer

# Load ticket dataset
df = pd.read_csv("../data/maintenance_tickets.csv")

# Load embedding model
model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

# Initialize ChromaDB
client = chromadb.PersistentClient(path="../chroma_db")

collection = client.get_or_create_collection(
    name="maintenance_tickets"
)

# Add tickets to vector database
for idx, row in df.iterrows():

    text = f"""
    Machine: {row['machine']}
    Department: {row['department']}
    Issue: {row['issue']}
    Root Cause: {row['root_cause']}
    Solution: {row['solution']}
    """
    document = f"""
    Machine: {row['machine']}

    Department: {row['maintenance_team']}

    Issue: {row['issue']}

    Priority Level: {row['priority']}

    Dominant Signal: {row['dominant_signal']}

    Root Cause: {row['root_cause']}

    Recommended Solution:
    {row['solution']}
    """
    embedding = model.encode(text).tolist()

    collection.add(
        ids=[str(row["ticket_id"])],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "priority": row["priority"],
            "status": row["status"]
        }]
    )

print("Vector database created successfully.")