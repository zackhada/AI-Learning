# AI & ML Knowledge Base
*Last updated: February 2, 2026*

---

## How to Use This Document

This is your running knowledge base for AI/ML concepts. Structure:
- **Core Concept**: The essential idea
- **My Understanding**: Your current grasp (update as you learn)
- **Open Questions**: What you're still working through
- **Connections**: How it links to other concepts

Upload this doc to future Claude conversations to continue building on it.

---

# Fundamentals

## Attention Mechanism

**Core Concept**  
Attention allows a model to dynamically focus on relevant parts of the input when producing each output. Instead of compressing all input into a fixed-size vector, attention computes a weighted sum over all input positions, where weights are learned based on relevance.

**Key Formula**  
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```
- Q (Query): What am I looking for?
- K (Key): What do I contain?
- V (Value): What do I return if matched?
- √d_k: Scaling factor to prevent softmax saturation

**My Understanding**  
*(Add your notes here as you learn)*

**Open Questions**  
- 

**Connections**  
→ Self-attention, Multi-head attention, KV caching

---

## Self-Attention

**Core Concept**  
When Q, K, and V all come from the same sequence. Each token attends to every other token in the sequence, learning contextual relationships.

**Why It Matters**  
Unlike RNNs that process sequentially, self-attention sees all positions simultaneously — enabling parallelization and capturing long-range dependencies.

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

**Connections**  
→ Attention mechanism, Transformer architecture

---

## Multi-Head Attention

**Core Concept**  
Instead of one attention function, run multiple in parallel with different learned projections. Each "head" can learn different types of relationships (syntactic, semantic, positional, etc.).

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

**Connections**  
→ Attention, Transformer blocks

---

## Positional Encoding

**Core Concept**  
Transformers have no inherent sense of order. Positional encodings inject sequence position information so the model knows token order.

**Types**  
- **Sinusoidal** (original): Fixed mathematical function
- **Learned**: Trained embedding per position
- **RoPE** (Rotary): Encodes relative position in attention computation
- **ALiBi**: Adds linear bias based on distance in attention scores

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

**Connections**  
→ Transformer architecture, Context length

---

## Layer Normalization

**Core Concept**  
Normalizes activations across the feature dimension (not batch). Stabilizes training by reducing internal covariate shift.

**Pre-LN vs Post-LN**  
- Post-LN (original): Normalize after residual connection
- Pre-LN (modern): Normalize before attention/FFN — more stable training

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

---

## Feed-Forward Network (FFN)

**Core Concept**  
The "thinking" layer in each transformer block. Two linear transformations with nonlinearity:

```
FFN(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

Typically expands dimensionality 4x then projects back down.

**Activation Functions**  
- ReLU (original)
- GELU (GPT-2+, BERT)
- SwiGLU (LLaMA, modern models) — gated variant

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

---

# Architectures

## Transformer (Original)

**Core Concept**  
Encoder-decoder architecture from "Attention Is All You Need" (2017). Encoder processes input, decoder generates output autoregressively.

**Key Components per Block**  
1. Multi-head self-attention
2. Add & Norm (residual + layer norm)
3. Feed-forward network
4. Add & Norm

**Variants**  
- Encoder-only: BERT (bidirectional, good for understanding)
- Decoder-only: GPT (causal, good for generation)
- Encoder-decoder: T5, original Transformer

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

---

## Mixture of Experts (MoE)

**Core Concept**  
Instead of one large FFN, have multiple "expert" FFNs and a router that selects which experts process each token. Allows massive parameter counts with lower compute (sparse activation).

**Key Components**  
- **Experts**: Independent FFN modules
- **Router/Gating**: Learned function that assigns tokens to experts
- **Top-k Selection**: Typically route to top 1-2 experts per token

**Trade-offs**  
- Pro: Scale parameters without proportional compute increase
- Con: Load balancing challenges, memory for all experts, routing overhead

**Examples**  
- Mixtral 8x7B: 8 experts, 2 active per token
- GPT-4 (rumored): MoE architecture
- Switch Transformer: Extreme sparsity (1 expert per token)

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

**Connections**  
→ FFN, Scaling laws, Sparse models

---

## State Space Models (SSMs)

**Core Concept**  
Alternative to attention that processes sequences through learned state transitions. Linear complexity in sequence length (vs quadratic for attention).

**Key Models**  
- **Mamba**: Selective state spaces, competitive with transformers
- **S4**: Structured state space for long sequences

**Trade-offs**  
- Pro: O(n) instead of O(n²), better for very long sequences
- Con: May lose some in-context learning capability

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

---

# Scaling & Training

## Scaling Laws

**Core Concept**  
Performance improves predictably with compute, data, and parameters. Discovered empirical relationships guide resource allocation.

**Chinchilla Laws**  
Optimal training: tokens ≈ 20× parameters. Previous models were undertrained on data.

**Key Insight**  
Given fixed compute budget, there's an optimal model size and data amount. Larger isn't always better if data-starved.

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

---

## KV Caching

**Core Concept**  
During autoregressive generation, cache the K and V projections from previous tokens. Only compute Q for new token, reuse cached K/V.

**Why It Matters**  
Without caching: O(n²) compute per token generated
With caching: O(n) per token (still accumulates, but much faster)

**Memory Trade-off**  
Cache grows with sequence length and batch size — often the bottleneck for long-context inference.

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

**Connections**  
→ Attention, Inference optimization

---

# Applications & Enterprise AI

## Agentic AI Architectures

**Core Concept**  
AI systems that can take actions, use tools, and operate autonomously to accomplish goals. Goes beyond single-turn Q&A.

**Key Patterns**  
- ReAct: Reasoning + Acting loop
- Tool use: Function calling, API integration
- Memory: Short-term (context) + long-term (retrieval)
- Planning: Task decomposition, self-correction

**Enterprise Considerations**  
*(Add Snowflake-specific notes on Cortex Agents here)*

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

---

## RAG (Retrieval-Augmented Generation)

**Core Concept**  
Combine retrieval from a knowledge base with generation. Grounds model responses in specific documents/data.

**Components**  
1. Embedding model (vectorize documents + queries)
2. Vector store (similarity search)
3. Generation model (synthesize retrieved context)

**Enterprise Patterns**  
*(Add Snowflake Cortex Search notes here)*

**My Understanding**  
*(Add your notes here)*

**Open Questions**  
- 

---

# Learning Log

## February 2026

### Feb 2
- Created this knowledge base
- Topics to dive deeper: *(add as you go)*

---

# Resources & References

**Papers**  
- "Attention Is All You Need" (2017) — Original transformer
- "Scaling Laws for Neural Language Models" (2020) — OpenAI scaling laws
- "Training Compute-Optimal Large Language Models" (2022) — Chinchilla paper

**Courses**  
- *(Add courses you're taking)*

**Books**  
- *(Add relevant books)*

---

*To continue building: Upload this file to a new Claude conversation and say "Let's add to my AI knowledge base" or ask about a specific concept.*
