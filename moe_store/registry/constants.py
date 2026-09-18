from transformers import (
    DbrxForCausalLM,
    DeepseekV2ForCausalLM,
    DeepseekV3ForCausalLM,
    GptOssForCausalLM,
    JambaForCausalLM,
    MixtralForCausalLM,
    NllbMoeForConditionalGeneration,
    OlmoeForCausalLM,
    OPTForCausalLM,
    PretrainedConfig,
    Qwen3MoeForCausalLM,
)

try:
    from transformers import DeepseekV4ForCausalLM
except ImportError:
    DeepseekV4ForCausalLM = None

try:
    from transformers import DeepseekV41ForCausalLM
except ImportError:
    DeepseekV41ForCausalLM = None

try:
    from transformers import Qwen3_5MoeForConditionalGeneration
except ImportError:
    Qwen3_5MoeForConditionalGeneration = None

try:
    from transformers import GlmMoeDsaForCausalLM
except ImportError:
    GlmMoeDsaForCausalLM = None

try:
    from transformers import Glm5NextForConditionalGeneration
except ImportError:
    Glm5NextForConditionalGeneration = None

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except ImportError:
    Qwen3VLMoeForConditionalGeneration = None

try:
    from transformers import Qwen3OmniMoeForConditionalGeneration
except ImportError:
    Qwen3OmniMoeForConditionalGeneration = None

try:
    from transformers import MiniMaxM3SparseForConditionalGeneration
except ImportError:
    MiniMaxM3SparseForConditionalGeneration = None

MODEL_MAPPING_NAMES = {
    "nllb": NllbMoeForConditionalGeneration,
    "mixtral": MixtralForCausalLM,
    "opt": OPTForCausalLM,
    "deepseek_v3": DeepseekV3ForCausalLM,
    "deepseek": DeepseekV2ForCausalLM,
    "gptoss": GptOssForCausalLM,
    "qwen3": Qwen3MoeForCausalLM,
    "dbrx": DbrxForCausalLM,
    "olmoe": OlmoeForCausalLM,
    "jamba": JambaForCausalLM,
}

MODEL_MAPPING_TYPES = {
    "nllb": 2,
    "mixtral": 4,
    "opt": 3,
    "deepseek_v3": 5,
    "deepseek": 5,
    "gptoss": 6,
    "qwen3": 5,
    "dbrx": 4,
    "olmoe": 5,
    # JambaMLP registers gate_proj, up_proj, down_proj (not w1, w2, w3).
    "jamba": 5,
}

# DeepSeek-V4 support depends on a transformers build that ships
# DeepseekV4ForCausalLM. Only register it when the class is importable so the
# registry never maps an architecture to a None class.
if DeepseekV4ForCausalLM is not None:
    MODEL_MAPPING_NAMES["deepseekv4"] = DeepseekV4ForCausalLM
    MODEL_MAPPING_TYPES["deepseekv4"] = 5

# DeepSeek-V4.1 uses the V4 expert family. Register only when transformers
# provides the modeling class, mirroring the DeepSeek-V4 guard above.
if DeepseekV41ForCausalLM is not None:
    MODEL_MAPPING_NAMES["deepseekv41"] = DeepseekV41ForCausalLM
    MODEL_MAPPING_TYPES["deepseekv41"] = 5

# Qwen3.5-MoE (arch "Qwen3_5MoeForConditionalGeneration") uses per-expert
# gate_proj/up_proj/down_proj weights (expert-type 5, like Qwen3/DeepSeek); the
# v5 packed checkpoint tensors are expanded to per-expert on load. Registered
# only when the HF class is importable (mirrors the V4 guard above).
if Qwen3_5MoeForConditionalGeneration is not None:
    MODEL_MAPPING_NAMES["qwen3_5"] = Qwen3_5MoeForConditionalGeneration
    MODEL_MAPPING_TYPES["qwen3_5"] = 5

# GLM-MoE-DSA (arch "GlmMoeDsaForCausalLM") uses per-expert gate_proj/up_proj/
# down_proj weights (expert-type 5, like Qwen3/DeepSeek). Requires transformers
# >= 5.12 which ships GlmMoeDsaForCausalLM. Registered only when the HF class
# is importable (mirrors the V4 and Qwen3.5 guards above).
if GlmMoeDsaForCausalLM is not None:
    MODEL_MAPPING_NAMES["glmmoedsa"] = GlmMoeDsaForCausalLM
    MODEL_MAPPING_TYPES["glmmoedsa"] = 5

# GLM-5.3-Flash (arch "Glm5NextForConditionalGeneration", model_type
# "glm5_next") nests its MoE fields under text_config and routes 288
# per-expert gate_proj/up_proj/down_proj experts with the same sigmoid/
# noaux_tc router family as GlmMoeDsa (expert-type 5). The hybrid KDA
# linear-attention layers, DSA indexer, mHC hyper-connections, and vision
# tower stay resident. Requires transformers >= 5.16 which ships the class;
# registered only when it is importable (mirrors the guards above).
if Glm5NextForConditionalGeneration is not None:
    MODEL_MAPPING_NAMES["glm5next"] = Glm5NextForConditionalGeneration
    MODEL_MAPPING_TYPES["glm5next"] = 5

# Qwen3-VL-MoE (arch "Qwen3VLMoeForConditionalGeneration") nests its MoE
# fields under text_config and ships v5 batched expert tensors under
# `model.language_model.*`; experts expand to per-expert gate_proj/up_proj/
# down_proj (expert-type 5). The vision tower (`model.visual.*`) stays
# resident. Registered only when the HF class is importable (mirrors the
# guards above).
if Qwen3VLMoeForConditionalGeneration is not None:
    MODEL_MAPPING_NAMES["qwen3vlmoe"] = Qwen3VLMoeForConditionalGeneration
    MODEL_MAPPING_TYPES["qwen3vlmoe"] = 5

# Qwen3-Omni-MoE (arch "Qwen3OmniMoeForConditionalGeneration") nests its
# offloadable MoE under thinker_config.text_config with per-expert
# gate_proj/up_proj/down_proj weights (expert-type 5). The talker MoE,
# vision tower, and code2wav stacks stay resident. Registered only when
# the HF class is importable (mirrors the guards above).
if Qwen3OmniMoeForConditionalGeneration is not None:
    MODEL_MAPPING_NAMES["qwen3omnimoe"] = Qwen3OmniMoeForConditionalGeneration
    MODEL_MAPPING_TYPES["qwen3omnimoe"] = 5

# MiniMax-M3 (arch "MiniMaxM3SparseForConditionalGeneration", model_type
# "minimax_m3_vl") is a vision-language MoE that nests its MoE fields under
# text_config and ships v5 batched expert tensors under
# `model.language_model.*`; experts expand to per-expert gate_proj/up_proj/
# down_proj (expert-type 5). The vision tower (`model.vision_tower.*`), the
# multimodal projector, and the shared expert stay resident. Registered only
# when the HF class is importable (mirrors the guards above).
if MiniMaxM3SparseForConditionalGeneration is not None:
    MODEL_MAPPING_NAMES["minimaxm3"] = MiniMaxM3SparseForConditionalGeneration
    MODEL_MAPPING_TYPES["minimaxm3"] = 5


# Dense diffusion transformers (diffusers-style `_class_name` configs) have
# no MoE experts: every tensor is stored in dense groups and the expert type
# is DENSE_EXPERT_TYPE. Matching is substring-based on the lowered class
# name, like MODEL_MAPPING_NAMES keys.
DENSE_EXPERT_TYPE = 0
DENSE_MODEL_CLASSES = (
    "QwenImageTransformer2DModel",
    "MiniMaxH3Transformer3DModel",
)


def is_dense_architecture(architecture: str) -> bool:
    arch = architecture.lower()
    return any(cls.lower() in arch for cls in DENSE_MODEL_CLASSES)


def parse_expert_type(config: PretrainedConfig) -> int:
    architecture = (
        config.architectures[0].lower() if config.architectures else ""
    )
    if is_dense_architecture(architecture):
        return DENSE_EXPERT_TYPE
    arch = None
    # Match the most specific key first: "qwen3_5" and "deepseek_v3" both
    # contain shorter keys ("qwen3", "deepseek") as substrings, so longest-key
    # order prevents a shorter key from shadowing a more specific architecture.
    for supp_arch in sorted(MODEL_MAPPING_NAMES, key=len, reverse=True):
        if supp_arch in architecture:
            arch = supp_arch
            break
    if arch is None:
        raise RuntimeError(
            f"The `load_checkpoint_and_dispatch` function does not support the architecture {architecture}. "
            f"Please provide a model that is supported by the function. "
            f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}."
        )

    return MODEL_MAPPING_TYPES[arch]
