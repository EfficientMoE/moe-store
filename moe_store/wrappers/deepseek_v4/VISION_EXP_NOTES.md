# DeepSeek-V4-Flash-Vision-Exp weight-map delta inventory

Source of truth: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
`model.safetensors.index.json` (72,633 tensors) and `config.json`, fetched
2026-09-05. The checkpoint uses the **DeepSeek-native key scheme**
(`layers.N.*`, `embed.weight`, `head.weight`), which the existing V4-Flash
Path-B loader and DeepSeek-V4 parsing support. It does not use HF-style
`model.layers.*` names.

## Top-level prefix histogram

| Count | Prefix | Meaning |
|---|---|---|
| 43 x ~1570 | `layers.0` .. `layers.42` | text decoder layers; every layer carries 256 routed experts |
| 1569/1566/1573 | `mtp.0`, `mtp.1`, `mtp.2` | three nextn/MTP draft layers, including their own routed experts |
| 256 | `vision.blocks` | 32-block vision encoder |
| 1 + 2 | `vision.norm`, `vision.patch_embed` | vision encoder head/stem |
| 2 + 2 | `aligner.w1`, `aligner.w2` | vision-to-text aligner |
| 4 | `image_start`, `image_end`, `image_newline`, `image_pad` | learned image-token embeddings |
| 3 | `hc_head_base`, `hc_head_fn`, `hc_head_scale` | hyper-connection head, also present in V4-Flash |
| 1 each | `embed.weight`, `head.weight`, `norm.weight` | token embedding, LM head, final norm |

## Tensor classification

| Class | Match rule | Handling |
|---|---|---|
| `ROUTED_EXPERT` | `layers.<L>.ffn.experts.<E>.` with `L < num_hidden_layers` | text-layer expert host store and streaming, unchanged from V4-Flash |
| `MTP_NEXTN` | `mtp.<i>.` prefix | resident only when nextn layers are used; skipped for text-only serving |
| `RESIDENT_VISION` | `vision.`, `aligner.`, or an image-token key | resident vision-side tensor |
| `RESIDENT_TEXT` | all other tensors | existing non-expert resident load path |

Every decoder layer, including layer 0, is MoE in this checkpoint. The MTP
layers contain their own `ffn.experts.*` tensors, so classification must check
the `mtp.` prefix rather than infer MTP from a text layer index. This keeps MTP
experts out of the text-layer host store.

`is_vision_exp_config` uses `vision_n_layers`, which is absent from base
V4-Flash. `num_nextn_predict_layers=3` is informational rather than the model
discriminator.

GPU/e2e parity, checkpoint conversion passthrough validation, and DSpark
support remain out of scope for this classifier and Path-B text-only wiring.
