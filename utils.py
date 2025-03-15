import os
import subprocess
import numpy as np

import torch
from torchvision.transforms import Compose, Lambda, ToPILImage


def getData():
    os.makedirs("./data", exist_ok=True)

    if not os.path.exists("./data/sprites_1788_16x16.npy"):
        print("start download dataset")
        command = [
            "wget",
            "-P",
            "./data",
            "https://github.com/brain-xiang/8bit-diffusion-model/raw/main/data/sprites_1788_16x16.npy",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("download completed")
        else:
            print("download failed")

    data = np.load("./data/sprites_1788_16x16.npy", allow_pickle=True)
    return data


def corrupt(x, amount):
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # Sort shape so broadcasting works
    return x * (1 - amount) + noise * amount


imgToTensor = Compose(
    [
        Lambda(lambda t: torch.from_numpy(t).float()),
        Lambda(lambda t: (t / 127.5) - 1),
        Lambda(lambda t: t.permute(0, 3, 1, 2)),
    ]
)

tensorToImg = Compose(
    [
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),
        Lambda(lambda t: t * 255.0),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ]
)
