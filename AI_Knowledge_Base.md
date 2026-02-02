# AI & ML Knowledge Base
*Last updated: February 2, 2026*

---

This document tracks my actual understanding of AI/ML concepts, validated through self-assessment.

---

# Validated Understanding

## Core Fundamentals (Solid)

### Neural Networks
- Inputs → weights → output structure
- Weights randomly initialized, tuned by backprop
- Training sets weights, inference uses fixed weights

### Backpropagation
- Weights updated based on comparing predicted vs actual output
- **Gap**: Mechanism details (gradients, chain rule, gradient descent)

### Tokens
- Text encoded to numbers for matrix multiplication
- ~0.75 words per token, ~4 characters
- Words often split into subwords for vocabulary efficiency

### Embeddings
- Tokens become high-dimensional vectors encoding meaning
- Created during training
- Enable semantic similarity comparisons

### Training vs Inference
- Training: setting weights through learning
- Inference: using fixed weights to generate outputs

---

## Attention Mechanism (Partial)

### What I Know
- Q (query): what to look for
- K (key): what the vector contains
- V (value): information contributed when Q-K match well
- Multiple attention heads learn different relationships
- Q/K/V learned through backprop

### Gaps
- √d_k scaling purpose
- Softmax role in attention
- Self-attention vs cross-attention (Q/K/V same sequence vs different sequences)

---

## RAG (Retrieval-Augmented Generation)

### What I Know
- Architecture that combines retrieval + generation (not a model itself)
- Unstructured data → chunks → embeddings → vector store
- Query embedded, similar chunks retrieved, passed to LLM
- LLM generates answer grounded in retrieved context
- More flexible than fine-tuning, stays current with document updates

### Corrections Applied
- RAG doesn't verify truth — it grounds in YOUR documents (which could be wrong)
- Chunking matters for precision, not just context window limits
- Smaller chunks = more specific embeddings = better retrieval

### Gaps
- Chunk size optimization strategies
- Hybrid search (semantic + keyword) trade-offs
- Re-ranking techniques

---

## Vector Databases

### What I Know
- Store embeddings as high-dimensional vectors
- Optimized for similarity search (nearest neighbors)

### Learned This Session
- Use specialized algorithms (HNSW, IVF) for efficient search
- Different from regular DBs which use exact matching and B-tree indexes
- Snowflake Cortex Search handles this internally

---

## Search Types

### Semantic Search
- Matches based on meaning/similarity
- Uses vector embeddings

### Keyword Search
- Uses inverted indexes (word → documents containing it)
- NOT regex-based
- Better for: exact matches (SKUs, error codes), legal/compliance, ambiguous semantics

---

## Agents

### What I Know
- Have access to tools (APIs, MCPs, functions)
- Follow custom instructions
- Key distinction: the **loop** — observe → reason → act → repeat

### Learned This Session
- Simple LLM call is one-shot; agents iterate until goal is met
- Tools = functions to interact with external systems

---

## Snowflake Cortex

### What I Know
- Cortex Analyst: natural language to SQL
- Cortex Search: hybrid semantic + keyword search
- Cortex Agents: orchestrate multiple agents
- UDFs for custom functions
- Cortex LLM functions: COMPLETE(), SUMMARIZE() in SQL

---

## LLM Parameters

### Temperature
- Controls output variation/randomness
- **Low temperature** (→0): picks most probable tokens, deterministic
- **High temperature**: flattens distribution, more random/creative
- **Correction**: I had this backwards initially

### Context Window
- Maximum tokens model can process at once
- When exceeded, options: truncation, re-ranking, summarization

---

## Production AI Concepts

### Latency vs Throughput
- **Latency**: time for one request (prompt → response)
- **Throughput**: requests per second system can handle
- NOT about context window size

### Guardrails
- Constraints to keep AI safe and compliant
- Types: input (block injection), output (filter harmful content, prevent PII leakage), topic restrictions, human-in-the-loop

### Prompt Engineering
- Curating prompts for desired output
- Techniques: few-shot examples, chain-of-thought, role assignment

---

## Fine-tuning vs RAG

| Aspect | Fine-tuning | RAG |
|--------|-------------|-----|
| What changes | Model weights | Context provided |
| Data freshness | Frozen at training | Updates anytime |
| Flexibility | Less flexible | More flexible |
| Cost | Training cost | Retrieval infra |

---

# Concepts to Learn

## Priority (foundational gaps)
- [ ] Self-attention vs cross-attention
- [ ] Why √d_k scaling in attention
- [ ] Softmax role in attention
- [ ] Loss functions (cross-entropy, etc.)
- [ ] Gradient descent mechanics
- [ ] Why transformers beat RNNs (parallelization)

## Enterprise AI
- [ ] Chunk size optimization
- [ ] Hybrid search strategies
- [ ] Re-ranking techniques
- [ ] Prompt injection defense
- [ ] Evaluation metrics for RAG

## Advanced
- [ ] RLHF / alignment
- [ ] Fine-tuning techniques (LoRA, etc.)
- [ ] Scaling laws
- [ ] MoE architectures
- [ ] Inference optimization

---

# Learning Log

## February 2026

### Feb 2
- Reset knowledge base with validated self-assessment
- **Session 1 topics**: NN basics, backprop, tokens, embeddings, attention Q/K/V
- **Session 2 topics**: RAG, vector databases, agents, Cortex, temperature, guardrails
- **Corrections made**:
  - Temperature (had backwards — low = deterministic)
  - Keyword search uses inverted indexes, not regex
  - Throughput = requests/sec, not context size
  - RAG grounds in docs but doesn't verify truth

---
