"""
============================================================
Industrial Maintenance RAG System
Robust Retrieval Evaluation (LOGO-Style)
============================================================

Author:
Mustafa Demetgül

Project:
Agents-Maintenance-AI

Description:
------------------------------------------------------------
This script evaluates the semantic retrieval performance
of the industrial maintenance RAG system.

The project uses:
- Real predictive maintenance datasets
- AI-generated maintenance tickets
- Sentence Transformer embeddings
- ChromaDB vector database
- Local Llama3 reasoning

This evaluation focuses on the retrieval layer of the
RAG pipeline.

The goal is to measure how well the semantic search system
retrieves contextually relevant industrial maintenance
tickets for unseen machine models.

============================================================
Evaluation Strategy
============================================================

A LOGO-style evaluation strategy is applied:

    Leave-One-Model-Out (LOMO)

For each machine model:
- tickets belonging to that model are used as test queries
- semantic retrieval is performed using the vector database
- retrieved tickets are evaluated for relevance

This evaluates:
- semantic generalization capability
- robustness across unseen machine models
- retrieval consistency in industrial environments

============================================================
Evaluation Metrics
============================================================

1. Department Precision@K
------------------------------------------------------------
Measures whether retrieved tickets belong to the same
industrial department as the query ticket.

Examples:
- Mechanical
- Hydraulic
- Electrical
- Automation

Higher values indicate stronger semantic consistency.

------------------------------------------------------------

2. Priority Precision@K
------------------------------------------------------------
Measures whether retrieved tickets have similar priority
levels.

Examples:
- High
- Medium
- Low

This evaluates retrieval relevance in terms of operational
criticality.

------------------------------------------------------------

3. Root Cause Precision@K
------------------------------------------------------------
Measures whether retrieved tickets contain similar
maintenance root-cause information.

Examples:
- Bearing wear
- Cooling failure
- Hydraulic leakage
- Electrical instability

This metric evaluates industrial maintenance reasoning
consistency.

============================================================
Why This Evaluation Is Important
============================================================

Unlike traditional classification systems, RAG systems are
evaluated based on:

- retrieval relevance
- semantic consistency
- grounding quality
- contextual similarity
- robustness across unseen conditions

This evaluation framework provides a practical industrial
benchmark for semantic maintenance retrieval systems.

============================================================
Expected Outputs
============================================================

Example output:

------------------------------------------------------------
LOGO-STYLE RETRIEVAL EVALUATION RESULTS
------------------------------------------------------------

Department Precision@3: 0.91
Priority Precision@3: 0.93
Root Cause Precision@3: 0.88

Total Queries Evaluated: 761

------------------------------------------------------------

Higher scores indicate better semantic retrieval quality
and stronger industrial context awareness.

============================================================
"""
import chromadb
import pandas as pd

from sentence_transformers import SentenceTransformer

# =========================================================
# Load embedding model
# =========================================================

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================================
# Load vector database
# =========================================================

client = chromadb.PersistentClient(
    path="../chroma_db"
)

collection = client.get_collection(
    name="maintenance_tickets"
)

# =========================================================
# Load datasets
# =========================================================

tickets_df = pd.read_csv(
    "../data/maintenance_tickets.csv"
)

machines_df = pd.read_csv(
    "../data/PdM_machines.csv"
)

# =========================================================
# Merge machine model information
# =========================================================

tickets_df["machineID"] = (
    tickets_df["machine"]
    .str.extract(r'_(\d+)$')
    .astype(int)
)

tickets_df = tickets_df.merge(
    machines_df[["machineID", "model"]],
    on="machineID",
    how="left"
)

# =========================================================
# Evaluation settings
# =========================================================

TOP_K = 3

department_score = 0
priority_score = 0
rootcause_score = 0

total_queries = 0

# =========================================================
# LOGO-style evaluation
# Leave-One-Model-Out
# =========================================================

unique_models = tickets_df["model"].unique()

for test_model in unique_models:

    print(f"\nEvaluating unseen model: {test_model}")

    # Test set
    test_df = tickets_df[
        tickets_df["model"] == test_model
    ]

    for idx, row in test_df.iterrows():

        query = row["issue"]

        expected_department = row["department"]
        expected_priority = row["priority"]
        expected_root = row["root_cause"]

        # =================================================
        # Create embedding
        # =================================================

        query_embedding = model.encode(
            query
        ).tolist()

        # =================================================
        # Retrieval
        # =================================================

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K
        )

        retrieved_docs = results["documents"][0]

        dept_matches = 0
        priority_matches = 0
        root_matches = 0

        # =================================================
        # Analyze retrievals
        # =================================================

        for doc in retrieved_docs:

            doc_lower = doc.lower()

            # Department relevance
            if expected_department.lower() in doc_lower:
                dept_matches += 1

            # Root cause relevance
            if expected_root.lower() in doc_lower:
                root_matches += 1

            # Priority relevance
            if expected_priority.lower() in doc_lower:
                priority_matches += 1

        # =================================================
        # Precision@K
        # =================================================

        department_score += dept_matches / TOP_K

        priority_score += priority_matches / TOP_K

        rootcause_score += root_matches / TOP_K

        total_queries += 1

# =========================================================
# Final metrics
# =========================================================

department_precision = (
    department_score / total_queries
)

priority_precision = (
    priority_score / total_queries
)

rootcause_precision = (
    rootcause_score / total_queries
)

# =========================================================
# Print results
# =========================================================

print("\n")
print("=" * 60)
print("LOGO-STYLE RETRIEVAL EVALUATION RESULTS")
print("=" * 60)

print(
    f"Department Precision@{TOP_K}: "
    f"{department_precision:.2f}"
)

print(
    f"Priority Precision@{TOP_K}: "
    f"{priority_precision:.2f}"
)

print(
    f"Root Cause Precision@{TOP_K}: "
    f"{rootcause_precision:.2f}"
)

print(f"Total Queries Evaluated: {total_queries}")

print("=" * 60)