//! ## ZigFormer
//!
//! ZigFormer is an implementation of a transformer-based language model (like GPT-2)
//! written in pure Zig. It provides a framework for understanding and experimenting with
//! transformer architectures.
//!
//! ### Features
//!
//! - **Transformer Architecture**: Multi-head self-attention, feed-forward networks,
//!   layer normalization, and positional embeddings
//! - **Training**: Adam optimizer, learning rate scheduling, and gradient accumulation
//! - **Inference**: Greedy decoding, top-k/top-p sampling, and beam search
//! - **Performance**: SIMD-optimized matrix operations and KV caching
//! - **Educational**: Documentation with mathematical formulas and explanations
//!
//! ### Quickstart
//!
//! ```zig
//! const zigformer = @import("zigformer");
//! const std = @import("std");
//!
//! // Create a vocabulary
//! var vocab = zigformer.vocab.Vocab.init(allocator);
//! try vocab.build(&[_][]const u8{ "hello", "world", "</s>" });
//!
//! // Initialize model
//! var model = try zigformer.llm.LLM.init(allocator, vocab);
//! defer model.deinit();
//!
//! // Generate text
//! const output = try model.predict("hello");
//! defer allocator.free(output);
//! ```
//!
//! ### Module Organization
//!
//! - `linalg`: Matrix operations and linear algebra
//! - `vocab`: Vocabulary and tokenization
//! - `embeddings`: Token and positional embeddings
//! - `self_attention`: Multi-head self-attention mechanism
//! - `feed_forward`: Position-wise feed-forward networks
//! - `layer_norm`: Layer normalization
//! - `transformer`: Transformer block (attention and FFN)
//! - `output_projection`: Output layer for vocabulary prediction
//! - `llm`: Language model implementation with training and inference API
//! - `optimizer`: Adam optimizer
//! - `lr_scheduler`: Learning rate scheduling
//! - `parallel`: Parallel computation utilities
//! - `layer`: Neural network layer interface
//!
//! ### Architecture
//!
//! ```
//! Input Tokens
//!     ↓
//! Embeddings (token + positional)
//!     ↓
//! TransformerBlock × N
//!   ├─ LayerNorm
//!   ├─ Multi-Head Self-Attention
//!   ├─ Residual Connection
//!   ├─ LayerNorm
//!   ├─ Feed-Forward Network
//!   └─ Residual Connection
//!     ↓
//! Output Projection → Vocabulary Logits
//! ```
//!
//! ### References
//!
//! - "Attention Is All You Need" (Vaswani et al., 2017)
//! - "Language Models are Unsupervised Multitask Learners" (GPT-2, Radford et al., 2019)

const std = @import("std");

pub const config = @import("config");
pub const linalg = @import("zigformer/linear_algebra.zig");
pub const vocab = @import("zigformer/vocab.zig");
pub const optimizer = @import("zigformer/optimizer.zig");
pub const lr_scheduler = @import("zigformer/lr_scheduler.zig");
pub const parallel = @import("zigformer/parallel.zig");
pub const layer = @import("zigformer/layer.zig");
pub const embeddings = @import("zigformer/embeddings.zig");
pub const layer_norm = @import("zigformer/layer_normalization.zig");
pub const self_attention = @import("zigformer/self_attention.zig");
pub const feed_forward = @import("zigformer/feed_forward.zig");
pub const transformer = @import("zigformer/transformer.zig");
pub const output_projection = @import("zigformer/output_projection.zig");
pub const llm = @import("zigformer/llm.zig");

test {
    std.testing.refAllDecls(@This());
}
