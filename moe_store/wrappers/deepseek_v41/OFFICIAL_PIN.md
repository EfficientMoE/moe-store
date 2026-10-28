# DeepSeek-V4.1-Flash official reference pin

- Repo: `deepseek-ai/DeepSeek-V4.1-Flash`
- Pinned revision: `314113f8a7dadb388215396ba5fbee2c610a04d6`
- Reviewed files: `inference/model.py`, `inference/kernel.py`,
  `inference/engram.py`, `inference/vision.py`, `inference/generate.py`,
  `inference/convert.py` (2026-10-28 review)
- Download:
  `hf download deepseek-ai/DeepSeek-V4.1-Flash --revision 314113f8a7dadb388215396ba5fbee2c610a04d6 --include "inference/*" --local-dir <official-dir>`
  (repeat with `--include "encoding/*"` and the root `config.json` /
  `chat_template.jinja`; the CLI accepts one `--include` pattern per call).
  Current local copy: `/mnt/raid0nvme0/leyang/official-v41`.
- Security: reference code executes locally; only this reviewed, pinned
  revision may be used. Bumping the pin requires re-review.

## Verified reference API (`inference/model.py` @ pin)

- **Model build:** `M.ModelArgs(**json.load(open("inference/config.json")))`
  — field names equal the *inference* config keys (NOT the HF root
  `config.json`; e.g. `n_layers`, `moe_inter_dim`, `route_scale`). Model:
  `M.Transformer(args, tokenizer)` — the tokenizer is **required** when
  `engram_layer_ids` is non-empty (it feeds `NgramHashState`). Constructing
  `Transformer` sets module-level globals `world_size`, `rank`,
  `default_dtype` (module defaults: `world_size=1`, `rank=0`).
- **Checkpoint flow:** official `generate.py` loads
  `model{rank}-mp{ws}.safetensors` produced by `inference/convert.py`.
  The raw hub shards keep the same key names for mp1 (verified against the
  hub index); expert FP4 weights are stored `I8 [out, in/2]` and must be
  viewed as `torch.float4_e2m1fn_x2` on load
  (`OfficialExpertHostStore._to_device` already does this).
- **MoE:** `M.MoE.__init__(self, layer_id, args)`; attrs `gate`,
  `shared_experts`, `experts` (ModuleList, None off-rank), `dim`,
  `n_routed_experts`, `experts_start_idx`, `experts_end_idx`.
  `MoE.forward(x, image_mask=None)`; it calls
  `self.gate(x_flat, None if image_mask is None else image_mask.flatten())`.
- **Gate:** `Gate.forward(x, image_mask)` → `(weights, indices)`. Scores:
  `linear(x.float(), weight.float()) / gate_temp` then sqrtsoftplus
  (`gate_temp` defaults to 1.0 and is absent from the released config).
  Selection adds `torch.where(image_mask, bias_vl, bias)`; weights come from
  unbiased scores, normalized (+1e-20) when `norm_topk_prob` and `topk > 1`,
  then `* route_scale` (1.5). Matches `topk_route_v41`.
- **Expert:** `M.Expert(dim, inter_dim, dtype=None, swiglu_limit=0.0)` with
  `w1/w2/w3 = Linear(...)`; forward clamps `up` to `[-limit, limit]` and
  `gate` to `(-inf, limit]` before `silu(gate) * up` — matches the V4
  `_expert_forward` helpers. Routed experts are built with
  `dtype=torch.float4_e2m1fn_x2` when `args.expert_dtype == "fp4"`.
- **Linear/quant:** `M.linear(x, weight)` dispatches on `weight.dtype`
  (fp4 → `fp4_gemm`, fp8 → `fp8_gemm`, else `F.linear`); quantized `Linear`
  registers `self.scale` as a parameter AND attaches it as `weight.scale`,
  so resident loads must copy `*.scale` tensors onto the module scale param.
- **Blocks:** `model.layers` is the decoder ModuleList; MoE lives at
  `block.ffn` with `block.ffn.gate` present on every backbone layer.
  `Block.forward(h, start_pos, pre_mix, image_mask)` (hyper-connections
  thread `pre_mix` between layers).
- **Transformer.forward:** `forward(input_ids, start_pos=0, images=None,
  token_types=None)` → `(output_ids, logits, main_hidden)`; it *samples
  internally* with `args.temperature` (use the returned `logits` for greedy
  checks) and drives engram hashing + KV caches via `start_pos` increments
  (follow `generate.py`'s prefill/decode loop). `logits` covers the final
  position only (`[b, vocab]`, `ParallelHead(full_logits=False)`).
- **Runtime environment:** the whole process must run under
  `torch.set_default_dtype(torch.bfloat16)` (kernel outputs are allocated
  with the default dtype) and `torch.set_default_device(<cuda>)` after
  model load (cached helpers like `get_window_topk_idxs` build index
  tensors on the default device).
- **SM120 caveat:** the tilelang `sparse_attn` kernel requests ~141 KB of
  dynamic shared memory (h=64, d=512) and fails `cudaFuncSetAttribute` on
  workstation Blackwell (~99 KB opt-in); the adapter auto-substitutes the
  exact-math `sparse_attn_eager` on such devices.
- **Hub shard dtype:** packed FP4 expert weights load as int8 from the hub
  shards and must be viewed as `torch.float4_e2m1fn_x2` before the
  reference `linear()` dispatch (`V41ExpertHostStore` does this).
- **Engram:** `model.layers[L].engram` is `Engram` with
  `engram.embed = ParallelEngramEmbedding(rows, 256)`:
  `embed.weight fp8_e4m3 [rows, 256]`, `embed.scale ue8m0 [rows, 8]`
  (group 32), `embed.forward(indices)` does `F.embedding` on both +
  rowwise dequant to bf16. Lookup indices have shape
  `[B, L, n_hash_cols]` (n_hash_cols = (max_ngram−1) × n_heads = 24) —
  the offload shim must support N-D index tensors. Engram layers: 1 and 14.
- **dspark MTP:** `model.mtp` (3 × `DSparkBlock`, own 128-expert MoE via
  `args.get_moe_config(layer_id ≥ n_layers)`); nothing in the reference
  calls `forward_spec` during normal generation → skipping `mtp.*` weights
  is safe for generation, but `Transformer.__init__` still *builds* the MTP
  modules when `dspark_block_size > 0` (their experts also need the
  null-expert swap to avoid allocating 128 FP4 experts × 3 layers).
- **Vision:** built only when `vision_n_layers > 0`; text-only runs must
  either zero `vision_n_layers` in a copied config or accept resident
  vision weights (~small). `image_mask=None` keeps `bias` routing.
