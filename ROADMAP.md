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

- [x] Optimizer (Adam)
- [x] Gradient clipping
- [x] Cross-entropy loss
- [x] Training loop (pretraining and fine-tuning)
- [x] Learning rate scheduling
- [x] Model checkpointing (save and load)
- [x] Mini-batch training
- [x] Gradient accumulation

### Inference

- [x] Greedy decoding
- [x] KV caching
- [x] Top-k and top-p sampling
- [x] Beam search

### Usability

- [x] Command line interface
- [x] Multi-threading support
- [x] SIMD optimizations
- [x] Model loading from a checkpoint
- [x] Configuration file
- [x] Improved error handling and validation

### GUI

- [x] Web server
- [x] Web interface
- [x] Model loading from a checkpoint
- [x] Configuration file
- [x] Improved error handling and validation
- [x] Markdown rendering and syntax highlighting
- [x] Interactive sampling controls (Top-k and Top-p)
- [x] Dark and light mode toggle
- [x] Model statistics display
