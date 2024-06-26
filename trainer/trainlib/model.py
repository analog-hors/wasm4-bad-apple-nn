import torch
import trainlib.native as native

INPUT_RANGE = 1.0
WEIGHT_RANGE = 1.0

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        point_dims = native.point_dims()
        embeddings = native.embeddings()
        self.em = torch.nn.Embedding(embeddings, 32)
        self.l0 = torch.nn.Linear(point_dims + 32, 128)
        self.l1 = torch.nn.Linear(128, 112)
        self.l2 = torch.nn.Linear(112, 1)

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        em1 = embeddings.int()
        em2 = torch.clamp(em1 + 1, max=native.embeddings() - 1)
        res = (embeddings - em1.float()).reshape((-1, 1))
        em = torch.lerp(self.em(em1), self.em(em2), res)

        x = self.l0(torch.cat((x, em), dim=1))
        x = torch.clamp(x, 0, INPUT_RANGE)
        x = self.l1(x)
        x = torch.clamp(x, 0, INPUT_RANGE)
        x = self.l2(x)
        x = torch.nn.functional.sigmoid(x)
        return x

    def clip(self):
        self.apply(_clipper)

def _clipper(module: torch.nn.Module):
    if hasattr(module, "weight"):
        w = module.weight.data
        w = w.clamp(-WEIGHT_RANGE, WEIGHT_RANGE)
        module.weight.data = w
