import ctypes, numpy

_LIB: ctypes.CDLL | None = None

def init_native_lib(path: str):
    global _LIB

    if _LIB is not None:
        return
    
    _LIB = ctypes.cdll.LoadLibrary(path)

    _LIB.point_dims.argtypes = []
    _LIB.point_dims.restype = ctypes.c_uint64

    _LIB.encode_frame_points.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
    ]
    _LIB.encode_frame_points.restype = None

    _LIB.loader_new.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
    ]
    _LIB.loader_new.restype = ctypes.c_void_p

    _LIB.loader_fill_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    _LIB.loader_fill_batch.restype = None

    _LIB.loader_drop.argtypes = [
        ctypes.c_void_p,
    ]
    _LIB.loader_drop.restype = None

def point_dims() -> int:
    assert _LIB is not None
    return _LIB.point_dims()

def encode_frame_points(points: numpy.ndarray, width: int, height: int, frame: float):
    assert _LIB is not None
    assert width >= 0 and height >= 0
    assert points.dtype == ctypes.c_float
    assert points.shape == (width * height, point_dims())
    points_c_array = numpy.ctypeslib.as_ctypes(points)
    points_ptr = ctypes.cast(points_c_array, ctypes.POINTER(ctypes.c_float))
    _LIB.encode_frame_points(points_ptr, width, height, frame)

class Loader:
    _ptr: ctypes.c_void_p

    def __init__(self, path: str, seed: int):
        assert _LIB is not None
        self._ptr = _LIB.loader_new(path.encode("utf-8"), seed)
        if not self._ptr:
            raise ValueError("Failed to create new Loader")

    def fill_batch(self, points: numpy.ndarray, targets: numpy.ndarray):
        assert _LIB is not None
        assert self._ptr

        assert points.dtype == ctypes.c_float
        assert targets.dtype == ctypes.c_float
        assert len(points.shape) == 2 and points.shape[1] == point_dims()
        assert len(targets.shape) == 2 and targets.shape[1] == 1
        assert points.shape[0] == targets.shape[0]

        batch_size = points.shape[0]
        points_c_array = numpy.ctypeslib.as_ctypes(points)
        targets_c_array = numpy.ctypeslib.as_ctypes(targets)
        _LIB.loader_fill_batch(
            self._ptr,
            ctypes.cast(points_c_array, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(targets_c_array, ctypes.POINTER(ctypes.c_float)),
            batch_size,
        )

    def __enter__(self):
        pass

    def __exit__(self):
        self._drop()

    def __del__(self):
        self._drop()

    def _drop(self):
        assert _LIB is not None
        if self._ptr:
            _LIB.loader_drop(self._ptr)
            self._ptr = ctypes.c_void_p(None)