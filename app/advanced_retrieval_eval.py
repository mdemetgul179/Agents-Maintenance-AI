
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

2. Priority Precision@K
------------------------------------------------------------
Measures whether retrieved tickets have similar priority
levels.

3. Root Cause Precision@K
------------------------------------------------------------
Measures whether retrieved tickets contain similar
maintenance root-cause information.

4. Dominant Signal Precision@K
------------------------------------------------------------
Measures whether retrieved tickets contain similar
dominant anomaly signals.

Examples:
- vibration
- pressure
- rotation

5. Average Semantic Similarity
------------------------------------------------------------
Measures embedding-level semantic similarity between
queries and retrieved maintenance tickets.

6. Unique Retrieval Ratio
------------------------------------------------------------
Measures duplicate retrieval robustness.

Higher values indicate better retrieval diversity.

============================================================
"""

"""
============================================================
Industrial Maintenance RAG System
Robust Retrieval Evaluation (LOGO-Style)
============================================================

Author:
Mustafa Demetgül

Project:
Agents-Maintenance-AI

============================================================
"""

import numpy as np
import chromadb
import pandas as pd

from sentence_transformers import SentenceTransformer

# =========================================================
# Load embedding model
# =========================================================

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
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
dominant_signal_score = 0

similarity_score_total = 0
unique_retrieval_score = 0

total_queries = 0

# =========================================================
# Priority similarity groups
# =========================================================

priority_groups = {

    "low": [
        "low",
        "medium"
    ],

    "medium": [
        "low",
        "medium"
    ],

    "high": [
        "high",
        "critical"
    ],

    "critical": [
        "critical",
        "high"
    ]
}

# =========================================================
# Signal similarity groups
# =========================================================

signal_groups = {

    "vibration": [
        "vibration",
        "rotation"
    ],

    "rotation": [
        "rotation",
        "vibration"
    ],

    "pressure": [
        "pressure"
    ]
}

# =========================================================
# Root-cause similarity groups
# =========================================================

root_groups = {

    "Bearing wear": [

        "Bearing wear",
        "Mechanical imbalance",
        "Lubrication degradation"
    ],

    "Mechanical imbalance": [

        "Mechanical imbalance",
        "Bearing wear",
        "Lubrication degradation"
    ],

    "Lubrication degradation": [

        "Lubrication degradation",
        "Bearing wear",
        "Mechanical imbalance"
    ],

    "Pressure leakage": [

        "Pressure leakage",
        "Hydraulic valve degradation",
        "Hydraulic flow instability"
    ],

    "Hydraulic valve degradation": [

        "Hydraulic valve degradation",
        "Pressure leakage",
        "Hydraulic flow instability"
    ],

    "Hydraulic flow instability": [

        "Hydraulic flow instability",
        "Pressure leakage",
        "Hydraulic valve degradation"
    ],

    "Cooling system degradation": [

        "Cooling system degradation",
        "Thermal overload",
        "Cooling fan malfunction"
    ],

    "Thermal overload": [

        "Thermal overload",
        "Cooling system degradation",
        "Cooling fan malfunction"
    ],

    "Cooling fan malfunction": [

        "Cooling fan malfunction",
        "Cooling system degradation",
        "Thermal overload"
    ],

    "PLC communication instability": [

        "PLC communication instability",
        "Control signal degradation",
        "Electrical synchronization issue"
    ],

    "Control signal degradation": [

        "Control signal degradation",
        "PLC communication instability",
        "Electrical synchronization issue"
    ],

    "Electrical synchronization issue": [

        "Electrical synchronization issue",
        "PLC communication instability",
        "Control signal degradation"
    ]
}

# =========================================================
# LOGO-style evaluation
# =========================================================

unique_models = tickets_df["model"].unique()

for test_model in unique_models:

    print("\n" + "=" * 60)
    print(f"Evaluating unseen model: {test_model}")
    print("=" * 60)

    # =====================================================
    # Test set
    # =====================================================

    test_df = tickets_df[
        tickets_df["model"] == test_model
    ]

    # =====================================================
    # Iterate queries
    # =====================================================

    for idx, row in test_df.iterrows():

        print(f"Processing query {total_queries + 1}")

        query = row["issue"]

        expected_department = str(
            row["department"]
        ).lower()

        expected_priority = str(
            row["priority"]
        ).lower()

        expected_root = str(
            row["root_cause"]
        )

        expected_signal = str(
            row["dominant_signal"]
        ).lower()

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
            n_results=TOP_K,
            include=["documents", "distances"]
        )

        retrieved_docs = results["documents"][0]
        retrieved_distances = results["distances"][0]

        # =================================================
        # Counters
        # =================================================

        dept_matches = 0
        priority_matches = 0
        root_matches = 0
        signal_matches = 0

        unique_docs = set()

        # =================================================
        # Analyze retrievals
        # =================================================

        for doc in retrieved_docs:

            unique_docs.add(doc)

            doc_lower = doc.lower()

            # =============================================
            # Department relevance
            # =============================================

            if expected_department in doc_lower:

                dept_matches += 1

            # =============================================
            # Priority relevance
            # =============================================

            allowed_priorities = priority_groups.get(

                expected_priority,
                [expected_priority]
            )

            if any(
                p in doc_lower
                for p in allowed_priorities
            ):

                priority_matches += 1

            # =============================================
            # Root-cause relevance
            # =============================================

            allowed_roots = root_groups.get(

                expected_root,
                [expected_root]
            )

            if any(
                r.lower() in doc_lower
                for r in allowed_roots
            ):

                root_matches += 1

            # =============================================
            # Dominant signal relevance
            # =============================================

            allowed_signals = signal_groups.get(

                expected_signal,
                [expected_signal]
            )

            if any(
                s in doc_lower
                for s in allowed_signals
            ):

                signal_matches += 1

        # =================================================
        # Similarity metric
        # =================================================

        avg_similarity = np.mean(

            [1 - d for d in retrieved_distances]
        )

        similarity_score_total += avg_similarity

        # =================================================
        # Precision@K metrics
        # =================================================

        department_score += (
            dept_matches / TOP_K
        )

        priority_score += (
            priority_matches / TOP_K
        )

        rootcause_score += (
            root_matches / TOP_K
        )

        dominant_signal_score += (
            signal_matches / TOP_K
        )

        # =================================================
        # Duplicate retrieval robustness
        # =================================================

        unique_retrieval_score += (

            len(unique_docs) / TOP_K
        )

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

dominant_signal_precision = (

    dominant_signal_score / total_queries
)

average_similarity = (

    similarity_score_total / total_queries
)

unique_retrieval_ratio = (

    unique_retrieval_score / total_queries
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

print(
    f"Dominant Signal Precision@{TOP_K}: "
    f"{dominant_signal_precision:.2f}"
)

print(
    f"Average Semantic Similarity: "
    f"{average_similarity:.2f}"
)

print(
    f"Unique Retrieval Ratio: "
    f"{unique_retrieval_ratio:.2f}"
)

print(f"Total Queries Evaluated: {total_queries}")

print("=" * 60)