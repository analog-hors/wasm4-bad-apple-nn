import os, numpy, torch, ctypes
from PIL import Image
import trainlib.native as native
from trainlib.model import Model

native.init_native_lib("target/release/libtrainer_native.so")

FRAMES_DIR = "frames/"
DECODED_DIR = "decoded/"
FRAME_WIDTH = 80
FRAME_HEIGHT = 60

model = Model()
model.load_state_dict(torch.load("model.bin"))
model.eval()

for frame in os.listdir(DECODED_DIR):
    os.remove(os.path.join(DECODED_DIR, frame))

frames = len(os.listdir(FRAMES_DIR))
image = Image.new("L", (FRAME_WIDTH, FRAME_HEIGHT))
points = numpy.zeros((image.width * image.height, native.point_dims()), dtype=ctypes.c_float)
for i in range(frames):
    native.encode_frame_points(points, image.width, image.height, i / (frames - 1))
    embedding = native.encode_frame_embedding(i / (frames - 1))
    decoded = model(torch.tensor(points), torch.full((points.shape[0],), embedding))
    image.frombytes(bytes(round(n.item() * 255) for n in decoded))
    image.save(os.path.join(DECODED_DIR, f"{i + 1:04}.png"))
