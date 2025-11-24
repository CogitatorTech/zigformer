## ZigFormer Roadmap

This document outlines the features implemented in ZigFormer and the future goals for the project.

> [!IMPORTANT]
> This roadmap is a work in progress and is subject to change.

### Core Architecture
- [x] Tokenization (word-based)
- [x] Vocabulary building
- [x] Embedding layer (token + positional)
- [x] Multi-head self-attention
- [x] Feed-forward network
- [x] Layer normalization
- [x] Residual connections

### Training
- [x] Adam optimizer
- [x] Gradient clipping
- [x] Cross-entropy loss
- [x] Training loop (pre-training and fine-tuning)
- [ ] Model checkpointing (save and load)
- [ ] Learning rate scheduling

### Inference
- [x] Greedy decoding
- [x] KV caching
- [ ] Top-k and top-p sampling
- [ ] Beam search

### Usability
- [x] Command line interface
- [ ] Configuration file
- [ ] Multi-threading support
- [ ] SIMD optimizations
