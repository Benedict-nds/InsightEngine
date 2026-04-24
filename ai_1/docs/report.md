# RAG Chatbot System Report

## Introduction

This project implements a Retrieval-Augmented Generation (RAG) chatbot for Academic City.

The system allows users to query:
- Ghana election data
- 2025 national budget

The goal is to build a fully manual RAG pipeline without using pre-built frameworks.

---

## Part A: Data Engineering

### Data Cleaning
- CSV normalized into structured rows
- PDF text cleaned and normalized

---

### Chunking Strategy
- Fixed-size chunking with overlap
- Ensures context continuity

---

### Justification
- Small chunks lose context
- Large chunks reduce retrieval precision
- Overlap improves semantic continuity

---

## Part B: Custom Retrieval System

### Implementation
- Embedding: Sentence Transformers
- Vector storage: FAISS
- Top-k retrieval
- Similarity scoring

---

### Failure Case
Query:
"Who won the election 2020?"

System failed due to:
- lack of structured reasoning
- absence of explicit winner statement

---

### Improvement
Hybrid scoring introduced:
- structured bonus (year, region)
- numeric bonus
- money bonus
- keyword scoring

---

### Result
- Improved retrieval relevance
- Correct answers for structured queries

---

## Part C: Prompt Engineering

### Prompt Design
- Context injection
- Strict grounding
- Hallucination control

---

### Experiment
- Without strict prompt → hallucination
- With strict prompt → grounded responses

---

## Part D: Full Pipeline

Pipeline:

Query → Retrieval → Context → Prompt → LLM → Response

Features:
- logging at each stage
- retrieval transparency
- prompt visibility

---

## Part E: Evaluation

### Results

| Mode | Performance |
|------|-----------|
| Vector | Poor |
| Hybrid | Strong |

---

### Strengths
- improved retrieval accuracy
- strong grounding
- transparent system

---

### Limitations
- no aggregation capability
- weak on vague queries
- requires explicit context

---

## Part F: Architecture

The system uses a modular pipeline:

- ingestion
- embedding
- retrieval
- ranking
- generation

Hybrid retrieval improves performance over vector-only methods.

---

## Part G: Innovation

A domain-specific scoring function was introduced:

- structured_bonus
- numeric_signal_bonus
- transport_keyword_bonus
- money_signal_bonus

This significantly improved retrieval quality.

---

## Conclusion

The project demonstrates that:

- manual RAG systems can be highly effective
- hybrid retrieval improves performance
- structured reasoning remains a limitation

Future work includes:
- aggregation logic
- query understanding
- reasoning layers