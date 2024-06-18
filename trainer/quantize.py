import torch
import trainlib.native as native
from trainlib.model import Model
from trainlib.quantize import quantize

native.init_native_lib("target/release/libtrainer_native.so")

model = Model()
model.load_state_dict(torch.load("model.bin"))
model.eval()

with open("model.rs", "w+") as file:
    quantize(model, file)
