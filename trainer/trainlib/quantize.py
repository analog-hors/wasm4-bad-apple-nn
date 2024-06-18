import torch, math
from trainlib.model import Model, INPUT_RANGE, WEIGHT_RANGE
from typing import TextIO

INPUT_SCALE = math.floor(127 / INPUT_RANGE)
WEIGHT_SCALE = math.floor(127 / WEIGHT_RANGE)

def quantize(model: Model, out: TextIO):
    out.write(f"pub const INPUT_SCALE: i8 = {INPUT_SCALE};\n")    
    out.write(f"pub const WEIGHT_SCALE: i8 = {WEIGHT_SCALE};\n")    

    def quantized_weight_str(tensor: torch.Tensor) -> str:
        if len(tensor.shape) == 0:
            n = round(tensor.item() * WEIGHT_SCALE)
            return str(n)
        return f"[{','.join(quantized_weight_str(t) for t in tensor)}]"

    def quantized_bias_str(tensor: torch.Tensor) -> str:
        if len(tensor.shape) == 0:
            n = round(tensor.item() * INPUT_SCALE * WEIGHT_SCALE)
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
