import numpy, torch, ctypes
import trainlib.native as native

class BatchLoader:
    _loader: native.Loader
    _points: numpy.ndarray
    _embeddings: numpy.ndarray
    _targets: numpy.ndarray

    def __init__(self, path: str, seed: int, batch_size: int):
        self._loader = native.Loader(path, seed)
        self._points = numpy.zeros((batch_size, native.point_dims()), dtype=ctypes.c_float)
        self._embeddings = numpy.zeros(batch_size, dtype=ctypes.c_float)
        self._targets = numpy.zeros((batch_size, 1), dtype=ctypes.c_float)

    def load_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._loader.fill_batch(self._points, self._embeddings, self._targets)
        points = torch.from_numpy(self._points).to(device)
        embeddings = torch.from_numpy(self._embeddings).to(device)
        targets = torch.from_numpy(self._targets).to(device)
        return points, embeddings, targets

    def __enter__(self):
        self._loader.__enter__()

    def __exit__(self):
        self._loader.__exit__()

    def __del__(self):
        self._loader.__del__()
