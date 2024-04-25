import torch
import trainlib.native as native

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        point_dims = native.point_dims()
        self.l0 = torch.nn.Linear(point_dims, 64)
        self.l1 = torch.nn.Linear(64, 64)
        self.l2 = torch.nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l0(x)
        x = torch.nn.functional.mish(x)
        x = self.l1(x)
        x = torch.nn.functional.mish(x)
        x = self.l2(x)
        x = torch.nn.functional.sigmoid(x)
        return x
