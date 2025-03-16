import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from ddpm import DDPMSampler
from utils import getData, imgToTensor, tensorToImg
from model import Unet

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
model = Unet(
    dim=config["image_size"],
    channels=config["channel"],
    dim_mults=(
        1,
        2,
        4,
    ),
)
print(f"{sum([p.numel() for p in model.parameters()]) / 1e6:.2f} M parameters")

model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
sampler = DDPMSampler()


def p_losses(denoise_model, x_start, t, loss_type="l1"):
    noise = torch.randn_like(x_start)
    x_noisy = sampler.q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


# ---------- train ----------
losses = []
model.train()
batch_iterator = tqdm(range(config["max_iters"]))
for batch in batch_iterator:
    x = next(iter(train_dataloader))
    x = x.to(device)  # Data on the GPU
    # change tiemsteps is not suggested
    t = torch.randint(0, sampler.timesteps, (x.shape[0],), device=device).long()
    loss = p_losses(model, x, t, loss_type="huber")

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
samples = sampler.sample(
    model, image_size=config["image_size"], batch_size=25, channels=config["channel"]
)

# 创建一个 5x5 的子图网格
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# 遍历每张图片并在对应的子图中显示
for i in range(25):
    img = tensorToImg(torch.tensor(samples[-1][i]))
    axes[i // 5, i % 5].imshow(img)
    axes[i // 5, i % 5].axis("off")

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.1, hspace=0.1)

# 保存整个网格为一张图片
plt.savefig("model_predictions.png", bbox_inches="tight", pad_inches=0.1)
plt.close()  # 关闭图像以释放内存
