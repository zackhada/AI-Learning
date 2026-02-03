# AI & ML Knowledge Base
*Last updated: February 3, 2026*

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
- Uses gradient descent to determine how to update each weight

### Gradient Descent
- Algorithm to minimize loss by iteratively adjusting weights
- `new_weight = old_weight - learning_rate × gradient`
- We subtract because gradient points uphill (toward higher loss), we want downhill
- Learning rate controls step size (too small = slow, too large = overshoots)

### Gradient
- The gradient tells you: "How much would loss change if I nudge this weight, and in which direction?"
- Positive gradient → increasing weight increases loss → decrease the weight
- Negative gradient → increasing weight decreases loss → increase the weight
- Magnitude indicates sensitivity (gradient of 5 matters more than 0.001)

### Loss Function
- A number measuring how wrong the model is (lower = better)
- For LLMs: cross-entropy loss = `-log(probability of correct token)`
- Punishes low confidence in correct answer harshly (1% confidence → high loss)
- 99% confident in right answer → loss ≈ 0.01
- 1% confident in right answer → loss ≈ 4.6

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

## Transformer Architecture

### Transformer Block (Layer)
- Components: attention + MLP + layer norm + residual connections
- "70B parameters" = weights in MLP layers, attention projections (Wq, Wk, Wv, Wo), and embeddings
- "32 layers" = 32 stacked transformer blocks

### Residual Connections
- Output is `x + sublayer(x)` instead of just `sublayer(x)`
- Allows gradients to flow directly during backprop
- Prevents vanishing gradients in deep networks
- **Gap**: Had intuition but not exact mechanism

### Positional Encoding
- Position encoded into token vector before entering transformer
- Types: sinusoidal (fixed), learned, RoPE (rotates vectors by position)

### Why Transformers Beat RNNs
- RNNs process tokens sequentially (can't compute token 5 until 1-4 done)
- Transformers use attention to see all tokens at once → parallel training on GPUs

### Encoder vs Decoder
- **Encoder**: Sees all tokens bidirectionally (BERT) — good for understanding
- **Decoder**: Sees only previous tokens, causal (GPT, Claude) — good for generation
- Modern LLMs are decoder-only
- Encoders still used for embeddings in RAG systems

### Causal Masking
- Prevents decoder from seeing future tokens during training
- Triangular mask: position N can only attend to positions 1 through N
- Mechanism: add -infinity to attention scores for future positions
- Softmax turns e^(-infinity) = 0 → those positions contribute nothing
- Why not just "tell" the model? Neural nets are math — can only manipulate numbers, not give instructions
- During inference: mask still applied but redundant (future tokens don't exist yet)

### Next Token Prediction
- Training: predict next token, compare to actual, backprop
- Batches contain sequences at multiple positions, all computed in parallel
- Each position only sees tokens before it
- Objective: minimize cross-entropy loss between predicted and actual

### Softmax
- Converts vector to probability distribution (sums to 1)
- Formula: `softmax(x_i) = e^(x_i) / Σ e^(x_j)`
- Exponentials make larger values much larger relative to smaller ones
- Used in attention (weights) and output layer (token probabilities)
- **Correction**: Not just scaling to 0-1, it's exponential normalization

---

## Attention Mechanism

### What I Know
- Q (query): what to look for
- K (key): what the vector contains
- V (value): information contributed when Q-K match well
- Multiple attention heads learn different relationships
- Q/K/V learned through backprop

### Self vs Cross Attention (Learned)
- **Self-attention**: Q, K, V all from same sequence
- **Cross-attention**: Q from one sequence, K/V from another (e.g., decoder attending to encoder)

### Gaps
- √d_k scaling purpose (prevents softmax saturation — still need to internalize)

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

## LLM Parameters & Training

### Temperature
- Controls output variation/randomness
- **Low temperature** (→0): picks most probable tokens, deterministic
- **High temperature**: flattens distribution, more random/creative

### Context Window vs Training Data
- **Training data** (trillions of tokens): total text across many examples
- **Context window** (e.g., 128K): max tokens in single input/output
- Separate concepts — train on many short sequences, inference handles long ones
- Context window length is trained (model learns to handle it)

### Pre-training vs Fine-tuning
- **Pre-training**: Next token prediction on massive text corpus
- **Fine-tuning**: Adjust behavior with curated examples
- Types: instruction tuning, RLHF, SFT (supervised fine-tuning)

### Parameters
- Weights in: MLP layers (W1, W2, biases), attention projections (Wq, Wk, Wv, Wo), embeddings
- Stored as floating point numbers in GPU memory

### Inference Cost
- More parameters = more matrix multiplications per token
- Costs: GPU memory (load all parameters), compute (FLOPs), KV cache (grows with context)

---

## Production AI Concepts

### Latency vs Throughput
- **Latency**: time for one request (prompt → response)
- **Throughput**: requests per second system can handle

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
- [ ] Why √d_k scaling in attention
- [x] Causal masking mechanics
- [x] Loss functions (cross-entropy details)
- [x] Gradient descent mechanics

## Enterprise AI
- [ ] Chunk size optimization
- [ ] Hybrid search strategies
- [ ] Re-ranking techniques
- [ ] Prompt injection defense
- [ ] Evaluation metrics for RAG

## Advanced
- [ ] RLHF / alignment details
- [ ] Fine-tuning techniques (LoRA, etc.)
- [ ] Scaling laws
- [ ] MoE architectures
- [ ] Inference optimization (quantization, KV cache)

---

# Learning Log

## February 2026

### Feb 3
- **Session 1 topic**: Gradient descent and loss functions
- **Learned**: Gradient = sensitivity + direction for each weight, loss = wrongness measure, cross-entropy for LLMs
- **Validated understanding**: Correctly answered why we subtract gradient, which model has higher loss, what gradient tells you
- **Session 2 topic**: Causal masking
- **Learned**: Triangular mask adds -infinity to block future positions, softmax zeros them out
- **Key insight**: Neural nets are math — can't "tell" them anything, must manipulate numbers

### Feb 2
- Reset knowledge base with validated self-assessment
- **Session 1 topics**: NN basics, backprop, tokens, embeddings, attention Q/K/V
- **Session 2 topics**: RAG, vector databases, agents, Cortex, temperature, guardrails
- **Session 3 topics**: Transformer architecture, residual connections, positional encoding, RNNs vs transformers, next token prediction, softmax, parameters, inference cost, pre-training vs fine-tuning
- **Corrections made**:
  - Temperature (had backwards — low = deterministic)
  - Keyword search uses inverted indexes, not regex
  - Throughput = requests/sec, not context size
  - RAG grounds in docs but doesn't verify truth
  - Softmax is exponential normalization, not simple 0-1 scaling
  - Context window ≠ batch size during training
- **Concepts clarified**:
  - Self-attention vs cross-attention
  - Encoder vs decoder (and why decoder-only for LLMs)
  - Residual connection mechanism
  - Why transformers parallelize better than RNNs

---
