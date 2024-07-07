import torch
import trainlib.native as native

INPUT_RANGE = 1.0
WEIGHT_RANGE = 1.0

def _lerp_embedding(em: torch.nn.Embedding, time: torch.Tensor) -> torch.Tensor:
    index = time * em.weight.shape[0]
    index_0 = torch.clamp(index.int(), max=em.weight.shape[0] - 1)
    index_1 = torch.clamp(index_0 + 1, max=em.weight.shape[0] - 1)
    progress = (index - index_0.float()).reshape((-1, 1))
    return torch.lerp(em(index_0), em(index_1), progress)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        point_dims = native.point_dims()
        self.em0 = torch.nn.Embedding(820, 20)
        self.em1 = torch.nn.Embedding(205, 44)
        self.l0 = torch.nn.Linear(point_dims + 20 + 44, 128)
        self.l1 = torch.nn.Linear(128, 96)
        self.l2 = torch.nn.Linear(96, 1)

    def forward(self, point: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        em0 = _lerp_embedding(self.em0, time)
        em1 = _lerp_embedding(self.em1, time)
        x = torch.cat((point, em0, em1), dim=1)
        x = self.l0(x)
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
