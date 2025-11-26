<div align="center">
  <picture>
    <img alt="ZigFormer Logo" src="logo.svg" height="25%" width="25%">
  </picture>
<br>

<h2>ZigFormer</h2>

[![Tests](https://img.shields.io/github/actions/workflow/status/CogitatorTech/zigformer/tests.yml?label=tests&style=flat&labelColor=282c34&logo=github)](https://github.com/CogitatorTech/zigformer/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-007ec6?label=license&style=flat&labelColor=282c34&logo=open-source-initiative)](https://github.com/CogitatorTech/zigformer/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-read-blue?style=flat&labelColor=282c34&logo=read-the-docs)](https://CogitatorTech.github.io/zigformer/)
[![Zig Version](https://img.shields.io/badge/Zig-0.15.2-orange?logo=zig&labelColor=282c34)](https://ziglang.org/download/)
[![Release](https://img.shields.io/github/release/CogitatorTech/zigformer.svg?label=release&style=flat&labelColor=282c34&logo=github)](https://github.com/CogitatorTech/zigformer/releases/latest)

An educational transformer-based LLM in pure Zig

</div>

---

ZigFormer is a fully functional implementation of a transformer-based large language model (LLM) written in pure Zig.
It aims to provide a clean, easy-to-understand implementation of a modern LLM (similar to GPT-2) with no external ML frameworks.
ZigFormer is mainly made for learning how a conventional transformer-based LLM works under the hood.

### Motivation

Most language model implementations rely on heavy frameworks like PyTorch or TensorFlow, which abstract away the core mechanics of how these models actually work.
ZigFormer takes a different approach by implementing everything from scratch in Zig, making it ideal for:

- **Learning**: Understanding transformer architecture by reading clean, well-documented code
- **Experimentation**: Modifying and extending the model without framework constraints
- **Performance**: Leveraging Zig's speed and control for efficient training and inference
- **Transparency**: Seeing exactly how attention mechanisms, embeddings, and training loops work

By building everything from the ground up using only basic linear algebra operations, ZigFormer demystifies the "magic" behind large language models.

### Features

- Implements core transformer architecture with multi-head self-attention
- Supports both pretraining and instruction fine-tuning
- Provides multiple decoding strategies (like greedy and beam search)
- Includes a CLI for training and inference
- Has a web-based UI for interactive chatting with the model
- Supports model checkpointing and the use of configuration files

See the [ROADMAP.md](ROADMAP.md) for the list of implemented and planned features.

> [!IMPORTANT]
> ZigFormer is in early development, so bugs and breaking changes are expected.
> Please use the [issues page](https://github.com/CogitatorTech/zigformer/issues) to report bugs or request features.

---

### Getting Started

You can get started with ZigFormer by following the steps below.

#### Installation

```bash
git clone https://github.com/CogitatorTech/zigformer.git
cd zigformer
zig build
```

> [!NOTE]
> ZigFormer is developed and tested with Zig 0.15.2.

#### Training a Model

```bash
zig build run -- --save-model model.bin
```

This will:

1. Load the training datasets from `datasets/simple_dataset/`
2. First, build a vocabulary of tokens from the data
3. Pre-train the model on the pretraining examples (raw text)
4. Fine-tune the model on the instruction-following examples (question-answer pairs)
5. Save the trained model to `model.bin`

Training parameters can be given through a configuration file or CLI arguments.

```bash
# Example using a configuration file for training
zig build run -- --config my_config.json
```

**Sample CLI Configuration:**

```json
{
    "pretrain_path": "datasets/simple_dataset/pretrain.json",
    "train_path": "datasets/simple_dataset/train.json",
    "pre_epochs": 10,
    "chat_epochs": 10,
    "batch_size": 32,
    "accumulation_steps": 1,
    "pre_lr": 0.0005,
    "chat_lr": 0.0001,
    "save_model_path": "model.bin",
    "interactive": true
}
```

#### Using the Web UI

You can run the web-based UI to chat with the trained model:

```bash
zig build run-gui -- --load-model model.bin
```

The UI can be accessed at `http://localhost:8080` by default.

You can also provide a configuration file for the UI:

```bash
zig build run-gui -- --config gui_config.json
```

**Sample Web UI Configuration:**

```json
{
    "port": 8080,
    "host": "0.0.0.0",
    "pretrain_path": "datasets/simple_dataset/pretrain.json",
    "train_path": "datasets/simple_dataset/train.json",
    "load_model_path": "model.bin",
    "max_request_size": 1048576,
    "max_prompt_length": 1000,
    "timeout_seconds": 30
}
```

#### Viewing Help

View all available options for the CLI and web-based UI:

```bash
zig build run -- --help
zig build run-gui -- --help
```

---

### Documentation

You can find the full API documentation for the latest release of
ZigFormer [here](https://CogitatorTech.github.io/zigformer/).

Alternatively, you can use the `zig build docs` command to generate the API documentation for the current version from
the source code.
This will generate HTML documentation in the `zig-out/docs/` directory.

#### Configuration Reference

**CLI Options:**

- `pretrain_path` - Path to pretraining dataset (JSON array of strings)
- `train_path` - Path to instruction-tuning dataset (JSON array of strings)
- `pre_epochs` - Number of pretraining epochs (default: 10)
- `chat_epochs` - Number of instruction-tuning epochs (default: 10)
- `batch_size` - Training batch size (default: 32)
- `accumulation_steps` - Gradient accumulation steps (default: 1)
- `pre_lr` - Pretraining learning rate (default: 0.0005)
- `chat_lr` - Instruction-tuning learning rate (default: 0.0001)
- `save_model_path` - Path to save trained model
- `load_model_path` - Path to load existing model
- `interactive` - Enter interactive mode after training (default: true)

**Web UI Options:**

- `port` - Server port (default: 8080)
- `host` - Bind address (default: "0.0.0.0")
- `pretrain_path` - Path to pretraining dataset (for vocabulary building)
- `train_path` - Path to training dataset (for vocabulary building)
- `load_model_path` - Path to trained model checkpoint
- `max_request_size` - Maximum request size in bytes (default: 1048576)
- `max_prompt_length` - Maximum prompt length in characters (default: 1000)
- `timeout_seconds` - Request timeout in seconds (default: 30)

---

### Contributing

Contributions are always welcome!
See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### License

ZigFormer is licensed under the MIT License (see [LICENSE](LICENSE)).

### Acknowledgements

* The logo is from [SVG Repo](https://www.svgrepo.com/svg/357414/brain) with some modifications.
* This project uses the [Chilli](https://github.com/CogitatorTech/chilli) CLI framework.
