# moe-store

Checkpoint conversion, model registry, and the arch-aware tensor store for
[MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity).

moe-store owns everything that knows about model architectures:

- **Checkpoint conversion**: HuggingFace safetensors checkpoints are converted
  into the v2 offload store, an architecture-aware on-disk layout in which all
  tensors of one expert (and of each dense layer module) are stored
  consecutively, so the engine fetches any expert with **one read at
  `(offset, size)`**. See [docs/store-format-v2.md](docs/store-format-v2.md).
- **Model registry**: supported architectures, expert naming/slot-order
  tables, resident-weight policies.
- **Config parsing**: `parse_moe_param`, `parse_expert_id`,
  `parse_expert_dtype` for every supported architecture.
- **Model wrappers**: `Sync*` MoE block replacements that talk to the
  offloading engine exclusively through the `EngineHooks` protocol.
- **Store C++ sources** (`moe_store/csrc/store/`): the async-I/O read path and v2 index
  consumed by MoE-Infinity as a build-time source dependency.

MoE-Infinity depends on moe-store; moe-store never imports MoE-Infinity.

## Install

```bash
pip install moe-store
```

## Convert a checkpoint

```bash
moe-store convert deepseek-ai/DeepSeek-V2-Lite-Chat /ssd/stores/dsv2-lite
moe-store inspect /ssd/stores/dsv2-lite
```

## Layout guarantees (v2)

| Invariant | Meaning |
|---|---|
| G1 | a tensor group never straddles a partition file |
| G2 | group start offsets are 4096-byte aligned (O_DIRECT) |
| G3 | expert members keep arch slot order (e.g. `[gate, up, down]`), never sorted |
| G4 | quantized experts keep weights + scales/blocks in the same group |

## License

Apache-2.0. See [LICENSE](LICENSE).
