import torch
from trainlib.model import Model
from typing import TextIO

WEIGHT_CLIP_RANGE = 1.0
WEIGHT_QUANT_RANGE = 127
BIAS_CLIP_RANGE = 1.0
BIAS_QUANT_RANGE = 127

def _clipper(module: torch.nn.Module):
    if hasattr(module, "weight"):
        w = module.weight.data
        w = w.clamp(-WEIGHT_CLIP_RANGE, WEIGHT_CLIP_RANGE)
        module.weight.data = w
    if hasattr(module, "bias"):
        b = module.bias.data
        b = b.clamp(-BIAS_CLIP_RANGE, BIAS_CLIP_RANGE)
        module.bias.data = b

def clip_model(model: Model):
    model.apply(_clipper)

def quantize(model: Model, out: TextIO):
    out.write(f"pub const WEIGHT_CLIP_RANGE: f32 = {WEIGHT_CLIP_RANGE};\n")
    out.write(f"pub const WEIGHT_QUANT_RANGE: f32 = {WEIGHT_QUANT_RANGE}.0;\n")
    out.write(f"pub const BIAS_CLIP_RANGE: f32 = {BIAS_CLIP_RANGE};\n")
    out.write(f"pub const BIAS_QUANT_RANGE: f32 = {BIAS_QUANT_RANGE}.0;\n")

    def quantized_weight_str(tensor: torch.Tensor) -> str:
        if len(tensor.shape) == 0:
            n = round(tensor.item() / WEIGHT_CLIP_RANGE * WEIGHT_QUANT_RANGE)
            n = min(max(n, -WEIGHT_QUANT_RANGE), WEIGHT_QUANT_RANGE)
            return str(n)
        return f"[{','.join(quantized_weight_str(t) for t in tensor)}]"

    def quantized_bias_str(tensor: torch.Tensor) -> str:
        if len(tensor.shape) == 0:
            n = round(tensor.item() / BIAS_CLIP_RANGE * BIAS_QUANT_RANGE)
            n = min(max(n, -BIAS_QUANT_RANGE), BIAS_QUANT_RANGE)
            return str(n)
        return f"[{','.join(quantized_bias_str(t) for t in tensor)}]"

    def struct_str(name: str, fields: dict[str, str]) -> str:
        fields_str = ",".join(f"{k}:{v}" for k, v in fields.items())
        return f"{name}{{{fields_str}}}"

    def type_str(name: str, args: list[int]) -> str:
        return f"{name}<{','.join(str(a) for a in args)}>"

    def write_embeddding(name: str, l: torch.nn.Embedding):
        struct = struct_str("Embedding", {
            "weight": quantized_weight_str(l.weight),
        })
        type = type_str("Embedding", [
            l.num_embeddings,
            l.embedding_dim,
        ])
        out.write(f"pub static {name}: {type} = {struct};\n")

    def write_linear(name: str, l: torch.nn.Linear):
        struct = struct_str("Linear", {
            "weight": quantized_weight_str(l.weight),
            "bias": quantized_bias_str(l.bias),
        })
        type = type_str("Linear", [
            l.in_features,
            l.out_features,
        ])
        out.write(f"pub static {name}: {type} = {struct};\n")

    write_embeddding("EM", model.em)
    write_linear("L0", model.l0)
    write_linear("L1", model.l1)
    write_linear("L2", model.l2)
