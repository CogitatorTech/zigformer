<div align="center">
  <picture>
    <img alt="ZigFormer Logo" src="logo.svg" height="25%" width="25%">
  </picture>
<br>

<h2>ZigFormer</h2>

[![Tests](https://img.shields.io/github/actions/workflow/status/habedi/zigformer/tests.yml?label=tests&style=flat&labelColor=282c34&logo=github)](https://github.com/habedi/zigformer/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-007ec6?label=license&style=flat&labelColor=282c34&logo=open-source-initiative)](https://github.com/habedi/zigformer/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-view-blue?style=flat&labelColor=282c34&logo=read-the-docs)](https://habedi.github.io/zigformer/)
[![Examples](https://img.shields.io/badge/examples-view-green?style=flat&labelColor=282c34&logo=zig)](https://github.com/habedi/zigformer/tree/main/examples)
[![Zig Version](https://img.shields.io/badge/Zig-0.15.1-orange?logo=zig&labelColor=282c34)](https://ziglang.org/download/)
[![Release](https://img.shields.io/github/release/habedi/zigformer.svg?label=release&style=flat&labelColor=282c34&logo=github)](https://github.com/habedi/zigformer/releases/latest)

An educational transformer-based LLM implementation in pure Zig

</div>

ZigFormer is an implementation of a Transformer-based language model written in pure [Zig](https://ziglang.org/).  
It is inspired by [RustGPT](https://github.com/tekaratzas/RustGPT) and serves both as an educational project and a playground for experimenting
with transformers outside of heavy ML frameworks.

---

### What This Is

ZigFormer demonstrates how to build and train a simple LLM in pure Zig:

- Tokenization and vocabulary building  
- Embedding layers  
- Transformer blocks (multi-head self-attention + feed-forward)  
- Layer normalization  
- Output projection for vocabulary prediction  
- Training loop with Adam optimizer and gradient clipping  
- Interactive chat mode  

The goal is clarity and modularity — not raw performance (yet).  

---

### Development

```sh
# Run tests
zig build test

# Build optimized release
zig build -Drelease-fast
````

If you prefer Make:

```sh
# Install dev deps (Debian example)
sudo apt-get install make
make install-deps

# Run common tasks
make help
```

---

### Interactive Mode (after training)

```
> How do mountains form?
Mountains are formed through tectonic forces or volcanism over long geological time periods.
```

---

### Contributing

Contributions are always welcome!
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to make a contribution.

### License

ZigFormer is licensed under the MIT License (see [LICENSE](LICENSE)).

### Acknowledgements

* The logo is from [SVG Repo](https://www.svgrepo.com/svg/357414/brain) with some modifications.
