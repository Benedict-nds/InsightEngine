# System Architecture

## Overview

The system implements a fully manual Retrieval-Augmented Generation (RAG) pipeline without using frameworks such as LangChain or LlamaIndex.

The architecture is designed to be modular, explainable, and suitable for domain-specific retrieval over structured and semi-structured data.

---

## Pipeline Flow

User Query → Retrieval → Context Selection → Prompt Construction → LLM → Response

---

## Components

### 1. Data Sources
- Ghana Election Dataset (CSV)
- 2025 Budget Statement (PDF)

---

### 2. Ingestion and Chunking
- CSV: each row treated as a structured document
- PDF: text extracted and split into chunks
- Chunking uses:
  - fixed size windows
  - overlap to preserve context

---

### 3. Embedding
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Converts text chunks into dense vector representations

---

### 4. Vector Storage
- FAISS index used for similarity search
- Cosine similarity implemented using L2-normalized vectors

---

### 5. Retrieval

Two modes implemented:

#### Vector Retrieval
- Uses embedding similarity only
- Fast but lacks structured awareness

#### Hybrid Retrieval (Improved)
Combines:
- vector similarity
- keyword scoring
- domain-specific signals

---

### 6. Hybrid Scoring System

Final score is computed as: final_score = α * vector_score + β * keyword_score + structured_bonus + numeric_bonus + money_bonus

Where:

- **structured_bonus**
  - boosts matching year and region

- **numeric_signal_bonus**
  - prioritizes chunks with relevant numbers

- **transport_keyword_bonus**
  - boosts domain-specific keywords (roads, transport)

- **money_signal_bonus**
  - prioritizes financial values (GH¢, million, billion)

---

### 7. Prompt Construction

The system uses a strict prompt template:

- Injects retrieved context
- Prevents hallucination
- Forces grounded responses

---

### 8. LLM Generation
- Model: `gpt-4o-mini`
- Generates final answer using provided context

---

### 9. Logging System

Each query produces:
- retrieved chunks
- scores
- prompt metadata
- model response

Logs are stored as JSON for experiment tracking and evaluation.

---

## Design Justification

- Manual pipeline ensures full control and transparency
- Hybrid retrieval improves accuracy over vector-only methods
- Domain-specific scoring aligns retrieval with query intent
- Logging enables reproducibility and evaluation

---

## Key Insight

Vector similarity alone is insufficient for structured data.

Combining semantic retrieval with domain-aware scoring significantly improves performance.