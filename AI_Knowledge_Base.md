# AI & ML Knowledge Base
*Last updated: February 2, 2026*

---

This document tracks my actual understanding of AI/ML concepts, validated through self-assessment.

---

# Validated Understanding

## Solid Fundamentals

### Neural Networks
- Understand: inputs → weights → output structure
- Understand: weights randomly initialized, tuned by backprop
- Understand: training sets weights, inference uses fixed weights to compute outputs

### Backpropagation
- Understand: weights updated based on comparing predicted vs actual output
- **Gap**: Mechanism details (gradients, chain rule, gradient descent)

### Tokens
- Understand: text encoded to numbers for matrix multiplication
- **Gap**: Subword tokenization (words split into pieces for vocabulary efficiency)

### Layers & Depth
- Understand: more layers = more information encoded
- Understand: attention heads encode semantic relationships, MLPs encode knowledge
- Partial understanding of why depth helps

### Embeddings
- Understand: tokens become large vectors encoding meaning in many dimensions
- Understand: embedding model created during training

### Loss Function
- Understand concept: compare model output to expected, use for backprop
- **Gap**: Terminology (loss function, cross-entropy, etc.)

### Transformers
- Understand: work across modalities
- **Gap**: Why better than RNNs (parallelization, no sequential bottleneck, long-range dependencies)

---

## Attention Mechanism

### What I Know
- Q (query): what to look for
- K (key): what the vector contains
- V (value): returned when match occurs
- Multiple attention heads learn different relationships
- Q/K/V learned through backprop

### Gaps
- V is not "value after transformation" — it's the information contributed when Q-K match
- Don't understand √d_k scaling
- Don't understand softmax role in attention
- Don't understand self-attention vs cross-attention

---

# Concepts to Learn

## Priority (foundational gaps)
- [ ] Self-attention vs cross-attention
- [ ] Why √d_k scaling in attention
- [ ] Softmax in attention
- [ ] Loss functions (cross-entropy, etc.)
- [ ] Gradient descent mechanics
- [ ] Why transformers beat RNNs (parallelization)

## Next tier
- [ ] Positional encoding
- [ ] Layer normalization
- [ ] Feed-forward networks in transformers
- [ ] Multi-head attention mechanics

## Advanced (later)
- [ ] RLHF / alignment
- [ ] Fine-tuning (LoRA, etc.)
- [ ] Scaling laws
- [ ] MoE architectures
- [ ] Inference optimization

---

# Learning Log

## February 2026

### Feb 2
- Reset knowledge base
- Completed initial assessment
- Solid on: basic NN structure, backprop purpose, tokens, embeddings, training vs inference
- Gaps identified: attention details, loss functions, transformer advantages over RNNs

---
