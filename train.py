import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils import getData, imgToTensor, tensorToImg, corrupt
from model import BasicUNet

config = {
    "batch_size": 128,
    "max_iters": 700 * 10,  # 700 for a dataset loop
    "eval_interval": 700,
    "lr": 1e-3,
    "image_size": 16,
    "channel": 3,
}

# ---------- check device ----------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using device:", device)
if device == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(device)}")
    print(
        f"Device memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3} GB"
    )
else:
    print("NOTE: If you have a GPU, consider using it for training.")
device = torch.device(device)

# ---------- create dataseter ----------
data = getData()

dataset = imgToTensor(data)
train_dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)


# ---------- create model ----------
model = BasicUNet()
print(f"{sum([p.numel() for p in model.parameters()]) / 1e6:.2f} M parameters")

model.to(device)
loss_fn = nn.L1Loss()
opt = torch.optim.Adam(model.parameters(), lr=config["lr"])

# ---------- train ----------
losses = []
model.train()
batch_iterator = tqdm(range(config["max_iters"]))
for batch in batch_iterator:
    x = next(iter(train_dataloader))
    x = x.to(device)  # Data on the GPU
    noise_amount = torch.rand(x.shape[0]).to(device)  # Pick random noise amounts
    noisy_x = corrupt(x, noise_amount)  # Create our noisy x

    # Get the model prediction
    pred = model(noisy_x)

    # Calculate the loss
    loss = loss_fn(pred, x)

    # Backprop and update the params:
    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(loss.item())

    if (batch % config["eval_interval"] == 0 and batch != 0) or batch == config[
        "max_iters"
    ]:
        # Print our the average of the loss values for this epoch:
        avg_loss = sum(losses[-config["eval_interval"] :]) / config["eval_interval"]
        print(f"step {batch}: train loss {avg_loss:.4f}")

# ---------- show result ----------
# Fetch some data
x = next(iter(train_dataloader))
x = x[:8]  # Only using the first 8 for easy plotting

# Corrupt with a range of amounts
amount = torch.linspace(0, 1, x.shape[0])  # Left to right -> more corruption
noised_x = corrupt(x, amount)

# Get the model predictions
with torch.no_grad():
    preds = model(noised_x.to(device)).detach().cpu()

# Plot
fig, axs = plt.subplots(3, 8, figsize=(20, 7))
for i in range(8):
    axs[0, i].imshow(tensorToImg(x[i]))
    axs[0, i].set_title(f"Input {i}")
    axs[0, i].axis("off")

    axs[1, i].imshow(tensorToImg(noised_x[i]))
    axs[1, i].set_title(f"Corrupted {i}")
    axs[1, i].axis("off")

    axs[2, i].imshow(tensorToImg(preds[i]))
    axs[2, i].set_title(f"Prediction {i}")
    axs[2, i].axis("off")

plt.tight_layout()
# 保存图片
plt.savefig("model_predictions.png", dpi=300, bbox_inches="tight")
# 关闭图形以释放内存
plt.close(fig)
