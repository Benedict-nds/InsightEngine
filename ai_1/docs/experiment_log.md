# Experiment Logs

## Overview

This section documents manual experiments conducted to evaluate the RAG system.

Each experiment includes:
- query
- observed behavior
- analysis
- improvement (if applied)

---

## Experiment 1: Aggregation Failure Case

### Query
Which party got the most votes in Greater Accra in 2020?

---

### Observed Behavior
- System returned "insufficient information"
- Retrieved chunks contained:
  - correct year (2020)
  - relevant candidates
  - vote counts

---

### Analysis
- Each chunk represented a single row
- No aggregation across rows was performed
- The model could not determine the maximum vote

---

### Conclusion
The system is effective for direct fact retrieval but fails on aggregation tasks.

---

## Experiment 2: Retrieval Failure (Vector Only)

### Query
Who won the election 2020?

---

### Observed Behavior
- Retrieved irrelevant regions
- Low-vote candidates ranked equally with high-vote candidates
- Incorrect or failed answers

---

### Root Cause
- Vector similarity ignored:
  - vote importance
  - structured constraints

---

## Experiment 3: Hybrid Retrieval Improvement

### Query
Which party got the most votes in Greater Accra in 2020?

---

### Observed Behavior
- Correct region and year retrieved
- High-vote candidates ranked higher
- Correct answer: NDC

---

### Improvement Applied
Hybrid scoring system:
- structured bonus (year, region)
- numeric bonus
- keyword scoring

---

### Result
Significant improvement in retrieval accuracy

---

## Experiment 4: Budget Retrieval (Numeric Failure)

### Query
How much was allocated for roads in 2025?

---

### Before Improvement
- Retrieved chunks with:
  - km values
  - percentages
- Failed to retrieve monetary allocation

---

### Issue
Numeric scoring treated all numbers equally

---

## Experiment 5: Budget Retrieval (Money-Aware Fix)

### Improvement
Added money-aware scoring:
- detects GH¢, million, billion
- penalizes non-financial numeric chunks

---

### After Improvement
- Correct chunk retrieved:
  - GH¢2.81 billion (Ghana Road Fund)
- Answer generated correctly

---

## Key Findings

1. Vector-only retrieval is insufficient for structured data
2. Hybrid scoring significantly improves retrieval quality
3. RAG systems struggle with aggregation tasks
4. Domain-specific scoring improves alignment with query intent