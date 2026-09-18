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

Multimodal MoE checkpoints (Qwen3-VL-MoE, Qwen3-Omni-MoE) convert the same
way; resident towers (vision, audio, Omni talker) land in dense groups while
the routed experts get one group per `(layer, expert)`:

```bash
moe-store convert Qwen/Qwen3-VL-235B-A22B-Instruct /ssd/stores/qwen3-vl
moe-store convert Qwen/Qwen3-Omni-30B-A3B-Instruct /ssd/stores/qwen3-omni
```

Dense diffusion transformers (diffusers-style `_class_name` configs) are
stored as dense groups only. Single-component pipeline repos auto-descend
into `transformer/`; roots with two or more weight-bearing components
(MiniMax-H3: `transformer`, `transformer_ref`, `vae`, `audio_vae`,
`text_encoder`) convert every component into one store — tensor names are
prefixed `<component>.`, H3 transformer blocks split into a non-AdaLN group
plus an `.adaln` bundle group per block, and the other components get one
group per safetensors shard. `--subfolder` still selects a single variant
component:

```bash
moe-store convert Qwen/Qwen-Image-2512 /ssd/stores/qwen-image
moe-store convert MiniMaxAI/MiniMax-H3 /ssd/stores/minimax-h3
moe-store convert MiniMaxAI/MiniMax-H3 /ssd/stores/minimax-h3-fl2va \
    --subfolder FL2VA/transformer
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
