import torch
import trainlib.native as native
from trainlib.batch_loader import BatchLoader
from trainlib.model import Model

native.init_native_lib("target/release/libtrainer_native.so")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAMES_DIR = "frames/"
BATCH_SIZE = 2 ** 18
BATCHES = 5000
LOG_INTERVAL = 100
SEED = 0xd9e

torch.manual_seed(SEED)
batch_loader = BatchLoader(FRAMES_DIR, SEED, BATCH_SIZE)

model = Model().to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = lambda o, t: torch.mean(torch.abs(o - t) ** 2.2)
print(f"total parameters: {sum(p.numel() for p in model.parameters())}")

model.train()
running_loss = 0
running_count = 0
for batch_index in range(BATCHES):
    points, targets = batch_loader.load_batch(DEVICE)
    outputs = model(points)
    loss = loss_fn(outputs, targets)

    optim.zero_grad()
    loss.backward()
    optim.step()

    running_loss += loss.item() * points.shape[0]
    running_count += points.shape[0]
    if (batch_index + 1) % LOG_INTERVAL == 0:
        print(f"[{batch_index + 1}/{BATCHES}] loss: {running_loss / running_count}", flush=True)
        running_loss = 0
        running_count = 0

torch.save(model.state_dict(), "model.bin")
