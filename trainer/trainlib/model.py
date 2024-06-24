import torch
import trainlib.native as native

INPUT_RANGE = 1.0
WEIGHT_RANGE = 1.0

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        point_dims = native.point_dims()
        embeddings = native.embeddings()
        self.em0 = torch.nn.Embedding(embeddings, 20)
        self.em1 = torch.nn.Embedding(embeddings // 4, 44)

        self.l0 = torch.nn.Linear(point_dims + 20 + 44, 128)
        self.l1 = torch.nn.Linear(128, 96)
        self.l2 = torch.nn.Linear(96, 1)

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        em0_i0 = embeddings.int()
        em0_i1 = torch.clamp(em0_i0 + 1, max=native.embeddings() - 1)
        em0_p = (embeddings - em0_i0.float()).reshape((-1, 1))
        em0 = torch.lerp(self.em0(em0_i0), self.em0(em0_i1), em0_p)

        em1_i0 = (embeddings / 4).int()
        em1_i1 = torch.clamp(em1_i0 + 1, max=native.embeddings() // 4 - 1)
        em1_p = (embeddings / 4 - em1_i0.float()).reshape((-1, 1))
        em1 = torch.lerp(self.em1(em1_i0), self.em1(em1_i1), em1_p)

        x = self.l0(torch.cat((x, em0, em1), dim=1))
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
