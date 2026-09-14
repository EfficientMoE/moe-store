# moe-store on-disk format, version 2 (normative)

This document is the compatibility contract between the moe-store writer
(checkpoint conversion) and every reader (the MoE-Infinity engine's C++
`TensorStore`, the pure-Python reader in `moe_store.index`, and
`moe-store inspect`). Any change to this format requires a version bump and
an entry in this document's revision table.

## 1. Store directory

```
<store_dir>/
├── store_index          # binary index, format below
└── store_data_<N>       # partition files, N = 0,1,2,... contiguous
```

A store directory holds exactly one converted model. Partition files are
dense binary blobs with no headers; all structure lives in `store_index`.

## 2. Concepts

- **Tensor**: one weight tensor, identified by a `tensor_id` (uint32,
  unique within the store, assigned by the converter).
- **Group**: the read-once unit; one *topology node*. Either
  - an **expert group**: every tensor of one routed expert in one layer
    (weight matrices plus, for quantized checkpoints, their scale/block
    tensors), or
  - a **dense group**: every tensor of one dense (non-expert) module
    (e.g. one layer's attention block, an embedding, `lm_head`).
- **group_id** (uint64): `(high32 = group index within stage,
  low32 = stage index)`; the last stage uses `0xFFFFFFFF` as the high32
  end-of-pipeline marker. This mirrors the engine's `corr_id` convention.

A store MAY contain zero expert groups: dense models (e.g. diffusion
transformers such as Qwen-Image or MiniMax-H3) convert to dense groups
only, and readers MUST NOT assume at least one sparse stage exists.

## 3. Layout invariants

| # | Invariant (MUST) |
|---|---|
| G1 | A group never straddles a partition file: `offset + total_size <= partition_size` for its file. The writer rolls to a new partition before writing a group that would not fit. A single group larger than `partition_size` is a conversion error. |
| G2 | `offset % 4096 == 0` for every group start (O_DIRECT requirement). Members are packed back-to-back inside the group at the writer's member alignment; readers MUST accept any `rel_offset % 64 == 0`. The current writer emits 4096-byte member alignment so per-member strides equal the engine's 4KiB aligned-size math; a future writer may tighten to 64 without a format bump. |
| G3 | Members of an expert group appear in **architecture slot order** as defined by the registry (e.g. `[gate/w1, up/w3, down/w2]` for Mixtral-family experts). Readers MUST NOT sort members. Dense group members appear in the canonical order recorded in the index. |
| G4 | Quantized experts (FP8 block scales, FP4/MXFP4 blocks+scales, GPTQ/AWQ packed tensors) keep every member of one expert in one group, with each scale/aux tensor immediately following its weight tensor in slot order. |
| G5 | `tensor_id`s are assigned in group order, members in slot order: the k-th member of the g-th group has a strictly larger id than every tensor of groups 0..g-1. Ids are dense (0..N-1). |

Consequences: any group is fully readable with a single
`pread(file, dst, total_size, offset)`; a member view is
`dst + rel_offset` with `size` bytes.

## 4. `store_index` binary format

All integers little-endian. Strings are `uint16 length` + UTF-8 bytes.

### 4.1 Header

| field | type | value |
|---|---|---|
| magic | u8[8] | `"MOESTOR2"` |
| version | u32 | `2` |
| flags | u32 | reserved, `0` |
| partition_size | u64 | bytes, default `10737418240` (10 GiB) |
| model_type | string | HF `config.model_type` of the source checkpoint; for dense diffusion components (no `model_type` key) the diffusers `_class_name`, e.g. `QwenImageTransformer2DModel` |
| checkpoint_name | string | source checkpoint id/path (informational) |
| num_stages | u32 | number of pipeline stages |
| num_groups | u64 | total groups |
| num_tensors | u64 | total tensors |

### 4.2 Stage table (`num_stages` records, stage order)

| field | type | notes |
|---|---|---|
| name | string | topology stage name (module prefix) |
| is_sparse | u8 | 1 = expert stage |
| num_groups | u32 | groups in this stage |

### 4.3 Group table (`num_groups` records, group_id order)

| field | type | notes |
|---|---|---|
| group_id | u64 | §2 |
| kind | u8 | 0 = dense, 1 = expert |
| layer_id | i32 | sparse layer index, −1 for dense groups |
| expert_id | i32 | expert index within layer, −1 for dense |
| file_id | u32 | partition file |
| offset | i64 | group start, G2-aligned |
| total_size | i64 | one read covers all members |
| num_members | u32 | |

Each group record is immediately followed by its member records:

| field | type | notes |
|---|---|---|
| tensor_id | u32 | G5 ordering |
| name | string | original checkpoint tensor name |
| rel_offset | i64 | from group start, 64-aligned |
| size | i64 | exact payload bytes (unpadded) |
| dtype | string | canonical dtype token, §4.4 |
| ndim | u8 | |
| shape | i64 × ndim | |

### 4.4 Dtype tokens

`float32, float16, bfloat16, float8_e4m3fn, uint8, int8, int32, int64`.
Packed quantized payloads (MXFP4 blocks, GPTQ qweight, …) use the storage
dtype of their raw buffer (`uint8`/`int32`), not the logical dtype.

## 5. Reader requirements

- Readers MUST validate magic + version and reject anything but `2`.
- Readers MUST use `(file_id, offset, total_size)` for group fetches and
  MUST NOT issue per-member reads on the hot path.
- Readers MAY memory-map `store_index`; it is immutable after conversion.

## 6. Writer requirements

- The writer MUST verify G1–G5 after planning and abort on violation.
- Partition files MUST be written with payloads at exactly
  `offset + rel_offset`; gaps (alignment padding) are zero-filled.
- Conversion is atomic per store: the writer writes to
  `<store_dir>.tmp-<pid>` and renames on success.

## 7. Revision history

| version | change |
|---|---|
| 2 | initial group-aware format (this document) |
| 1 | legacy per-tensor `archer_index` + `archer_param_<N>` (MoE-Infinity ≤ PR1; readable only by pre-split releases) |
