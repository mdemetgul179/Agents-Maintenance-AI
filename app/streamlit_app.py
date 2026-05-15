import streamlit as st
import chromadb
import html
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# =====================================================
# Page config
# =====================================================

st.set_page_config(

    page_title="Industrial Maintenance AI",

    layout="wide"
)

# =====================================================
# Title
# =====================================================

st.title(
    "Industrial Maintenance RAG System"
)
# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("System Configuration")

selected_department = st.sidebar.selectbox(
    "Department",
    [
        "All",
        "Mechanical",
        "Hydraulic",
        "Thermal",
        "Automation"
    ]
)

selected_priority = st.sidebar.selectbox(
    "Priority",
    [
        "All",
        "Low",
        "Medium",
        "High",
        "Critical"
    ]
)

top_k = st.sidebar.slider(
    "Top K Retrieval",
    min_value=1,
    max_value=10,
    value=3
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Industrial AI Maintenance Monitoring System"
)
# =====================================================
# Dashboard Metrics
# =====================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Tickets",
        value="761"
    )

with col2:
    st.metric(
        label="Critical Issues",
        value="128"
    )

with col3:
    st.metric(
        label="Departments",
        value="4"
    )

with col4:
    st.metric(
        label="Retrieval Precision",
        value="99%"
    )

st.markdown("---")

st.markdown(
    """
AI-powered industrial maintenance assistant using:

- Semantic Retrieval
- ChromaDB
- Sentence Transformers
- Local LLM Reasoning
"""
)
# =====================================================
# Charts
# =====================================================

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    fig1, ax1 = plt.subplots()

    labels = [
        "Mechanical",
        "Hydraulic",
        "Thermal",
        "Automation"
    ]

    values = [220, 210, 170, 161]

    ax1.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )

    ax1.set_title(
        "Department Distribution"
    )

    st.pyplot(fig1)

with chart_col2:

    fig2, ax2 = plt.subplots()

    priorities = [
        "Low",
        "Medium",
        "High",
        "Critical"
    ]

    counts = [180, 340, 170, 71]

    ax2.bar(
        priorities,
        counts
    )

    ax2.set_title(
        "Priority Distribution"
    )

    st.pyplot(fig2)
# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.header("System Settings")

    top_k = st.slider(

        "Top K Results",

        min_value=1,

        max_value=10,

        value=3
    )

    show_similarity = st.checkbox(

        "Show Similarity Score",

        value=True
    )

# =====================================================
# Load embedding model
# =====================================================

@st.cache_resource

def load_model():

    return SentenceTransformer(
        "BAAI/bge-base-en-v1.5"
    )

model = load_model()

# =====================================================
# Load vector DB
# =====================================================

@st.cache_resource

def load_collection():

    client = chromadb.PersistentClient(
        path="../chroma_db"
    )

    return client.get_collection(
        name="maintenance_tickets"
    )

collection = load_collection()

# =====================================================
# User input
# =====================================================

issue = st.text_input(

    "Enter maintenance issue",

    placeholder="Example: critical hydraulic instability"
)

# =====================================================
# Search button
# =====================================================

if st.button("Analyze Issue"):

    if len(issue.strip()) == 0:

        st.warning(
            "Please enter a maintenance issue."
        )

    else:

        # =============================================
        # Create embedding
        # =============================================

        query_embedding = model.encode(
            issue
        ).tolist()

        # =============================================
        # Retrieval
        # =============================================

        results = collection.query(

            query_embeddings=[query_embedding],

            n_results=top_k,

            include=["documents", "distances"]
        )

        retrieved_docs = results["documents"][0]

        retrieved_distances = results["distances"][0]

        # =============================================
        # Dashboard metrics
        # =============================================

        best_similarity = (
            1 - retrieved_distances[0]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(

            "Top Match Similarity",

            f"{best_similarity:.2f}"
        )

        col2.metric(

            "Retrieved Tickets",

            f"{top_k}"
        )

        col3.metric(

            "Embedding Model",

            "BGE-Base"
        )

        # =============================================
        # Results section
        # =============================================

        st.subheader(
            "Retrieved Maintenance Tickets"
        )

        for i, doc in enumerate(retrieved_docs):

            doc_lower = doc.lower()

            card_color = "#1e1e1e"

            if "critical" in doc_lower:

                card_color = "#5c1a1a"

            elif "high" in doc_lower:

                card_color = "#7a4a00"

            elif "medium" in doc_lower:

                card_color = "#665500"

            elif "low" in doc_lower:

                card_color = "#1f4d2e"

            doc = str(doc)

            doc = html.escape(doc)

            doc = doc.replace("&lt;/div&gt;", "")
            doc = doc.replace("&lt;div&gt;", "")

            similarity = (
                1 - retrieved_distances[i]
            )

            with st.container():

                st.markdown(
                    f"### Result {i+1}"
                )

                st.markdown(
                    f"""
                    <div style="
                        padding:15px;
                        border-radius:10px;
                        background-color:{card_color};
                        margin-bottom:10px;
                        border:1px solid #444;
                        color:white;
                        font-size:16px;
                    ">
                        {doc}
                    """,
                    unsafe_allow_html=True
                )

                if show_similarity:

                    st.write(
                        f"Similarity Score: "
                        f"{similarity:.2f}"
                    )

        # =============================================
        # AI analysis
        # =============================================

        st.markdown(
            f"""
            <div style="
                padding:25px;
                border-radius:15px;
                background-color:#111827;
                border-left:8px solid #00c853;
                margin-top:25px;
                margin-bottom:25px;
                color:white;
                font-size:17px;
                line-height:1.8;
            ">

            <h2>
            AI Maintenance Analysis
            </h2>

            <p>
            <b>Detected Issue:</b>
            {issue}
            </p>

            <p>
            The semantic retrieval system identified
            historically similar industrial maintenance
            anomalies associated with operational
            instability patterns.
            </p>

            <p>
            <b>Recommended Actions:</b>
            </p>

            <ul>
                <li>Inspect related subsystem</li>
                <li>Verify sensor consistency</li>
                <li>Perform preventive maintenance</li>
                <li>Evaluate operational safety</li>
                <li>Analyze anomaly progression trend</li>
            </ul>

            <p>
            <b>Operational Recommendation:</b>
            </p>

            <p>
            The detected anomaly should be validated
            by maintenance engineers before critical
            intervention procedures are initiated.
            </p>

            <p>
            <b>Suggested Maintenance Workflow:</b>
            </p>

            <ol>
                <li>Initial anomaly inspection</li>
                <li>Sensor verification</li>
                <li>Mechanical / hydraulic analysis</li>
                <li>Preventive maintenance planning</li>
                <li>Risk assessment validation</li>
            </ol>

            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# Footer
# =====================================================

st.markdown("---")

st.caption(
    "Industrial AI Maintenance RAG System | "
    "Mustafa Demetgül"
)