import time, torch
import trainlib.native as native
from trainlib.batch_loader import BatchLoader
from trainlib.model import Model

native.init_native_lib("target/release/libtrainer_native.so")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAMES_DIR = "frames/"
BATCH_SIZE = 2 ** 18
BATCHES = 20000
LOG_INTERVAL = 100
SEED = 0xd9e

torch.manual_seed(SEED)
batch_loader = BatchLoader(FRAMES_DIR, SEED, BATCH_SIZE)

model = Model().to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr = 0.001)
loss_fn = lambda o, t: torch.mean(torch.abs(o - t) ** 2.2)
print(f"total parameters: {sum(p.numel() for p in model.parameters())}")

model.train()
running_start = time.time()
running_loss = 0
running_count = 0
for batch_index in range(BATCHES):
    points, embeddings, targets = batch_loader.load_batch(DEVICE)
    outputs = model(points, embeddings)
    loss = loss_fn(outputs, targets)

    optim.zero_grad()
    loss.backward()
    optim.step()

    running_loss += loss.item() * points.shape[0]
    running_count += points.shape[0]
    if (batch_index + 1) % LOG_INTERVAL == 0:
        now = time.time()
        loss = running_loss / running_count
        bps = LOG_INTERVAL / (now - running_start)
        print(f"[{batch_index + 1}/{BATCHES}] loss: {loss}, {bps:.2f} batches/second", flush=True)

        running_start = now
        running_loss = 0
        running_count = 0

torch.save(model.state_dict(), "model.bin")
