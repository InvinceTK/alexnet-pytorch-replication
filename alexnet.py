# implement init weights
from torch.nn import Sequential, Conv2d, ReLU, LocalResponseNorm, Linear, MaxPool2d
from torch.nn.functional import max_pool3d
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import wandb
from dotenv import load_dotenv
import os
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import datasets
from PIL import Image
import einops

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2

#how to hf authenticate
#export HF_TOKEN="[token]"


def transform(img):
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((227)),
            v2.CenterCrop((227, 227)),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[1.0, 1.0, 1.0],
            ),
        ]
    )
    img = transforms.forward(img)
    return img

class AlexNet(nn.Module):
    """
    Neural network model consisting of layers propsed by AlexNet paper.
    """

    def __init__(self, num_classes=1000):
        """
        Define and allocate layers for this neural net.

        Args:
            num_classes (int): number of classes to predict with this model
        """
        super().__init__()
        # input size should be : (b x 3 x 227 x 227)
        # The image in the original paper states that width and height are 224 pixels, but
        # the dimensions after first convolution layer do not lead to 55 x 55.
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4
            ),  # (b x 96 x 55 x 55) 0
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)  | 8
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13) | 10
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13) | 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
        )
        # classifier is just a name for linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),  # 1
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),  # 4
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes),  # 6
        )
        self.init_weights()  # initialize bias

    def init_weights(self):
        # set weights and biases for net block
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layfers
        nn.init.constant_(self.net[4].bias, 1)
        nn.init.constant_(self.net[10].bias, 1)
        nn.init.constant_(self.net[12].bias, 1)

        # set weights and biases for the classifier head
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(self.classifier[1].bias, 1)
            nn.init.constant_(self.classifier[4].bias, 1)
            nn.init.constant_(self.classifier[6].bias, -7)

    def forward(self, x):
        i = 0
        for layer in self.net:
            x = layer(x)
            # print(f"Layer {i} | {str(layer)} | img.shape = {x.shape}")
            i += 1

        i = 0
        flatten = nn.Flatten()
        x = flatten(x)
        for layer in self.classifier:
            x = layer(x)
            # print(f"Layer {i} | {str(layer)} | img.shape = {x.shape}")
            i += 1
        return x

if __name__ == "__main__":
    model = AlexNet()

    # set hyperparams
    NUM_EPOCHS = 90
    BATCH_SIZE = 128
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e4
    LR = 1e4
    OUTPUT_PATH = "model.txt"

    # set optimiser, loss and dataloader, and transform dataset
    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    ds = ds.map(transform)
    train_dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), LR)

    # wandb init
    TEAM_ENTITY = "elliot-kaute"
    PROJECT = "alexnet-replication"

    # api-keys
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    HF_TOKEN=os.getenv("HF_TOKEN")

    # set hyperparams
    NUM_EPOCHS = 1
    BATCH_SIZE = 128
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e4
    LR = 1e4
    OUTPUT_PATH = "saved-models/alexnet.pt"

    # set optimiser, loss and dataloader
    train_dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), LR)

    # wandb init
    TEAM_ENTITY = "elliot-kaute"
    PROJECT = "alexnet-replication"

    with wandb.init(
        project="alexnet-replication", config={"lr": LR, "batch_size": BATCH_SIZE}
    ) as run:
        DEVICE = "cuda" if torch.cuda.is_available() else "mps"
        model.to(DEVICE)
        for epoch in range(NUM_EPOCHS):
            step = 0

            for batch in tqdm(iter(train_dl)):
                img = batch["image"]
                label = batch["label"]
                # print(img.shape, label.shape)
                img = img.to(DEVICE)
                label = label.to(DEVICE)

                # perform grad descent
                with torch.no_grad():
                    optimizer.zero_grad()
                    out = model.forward(img)
                    # tensor of 1000 outputs

                    loss = F.cross_entropy(out, label)
                    loss = loss_fn(out, label)

                    ## !! Important | if requires_grad not set on loss then loss.backward won't run
                    loss.requires_grad = True
                    loss.backward()

                    optimizer.step()

                if step % 5 == 0:
                    run.write_logs(f"five steps taken")
                    run.log(
                        {
                            "epoch": epoch,
                            "loss": loss,
                        }
                    )
                step += 1

    torch.save(model.state_dict(), OUTPUT_PATH)
