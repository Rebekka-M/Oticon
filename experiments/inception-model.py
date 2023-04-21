# %%
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import os
import sys

sys.path.insert(0, os.getcwd())

from data.loader import load_training, load_validation, n_freq, n_time, n_classes
from inception import FeatureConcatenation, InceptionA, InceptionB

# Set up wandb
# import api_key
import wandb

# os.environ["WANDB_API_KEY"] = api_key.key
WANDB__SERVICE_WAIT = 300



# %%

force_cpu = False

if t.cuda.is_available() and not force_cpu:
    device = t.device("cuda")
else:
    device = t.device("cpu")

print(device)

X_map = lambda X: t.from_numpy(X).to(dtype=t.float)[:, None, :, :]
y_map = lambda y: t.from_numpy(y).to(dtype=t.uint8)
loader_map = lambda data: DataLoader(
    dataset=data,
    batch_size=128,
    shuffle=True,
    num_workers=1,
    pin_memory="cuda" == device,
)


X_train, y_train = load_training()
X_val, y_val = load_validation()


X_train, X_val = map(X_map, (X_train, X_val))
y_train, y_val = map(y_map, (y_train, y_val))


data_train = TensorDataset(X_train, y_train)
data_val = TensorDataset(X_val, y_val)

loader_train, loader_val = map(loader_map, (data_train, data_val))


n_train = len(X_train)
n_val = len(X_val)

# %%

class Model(nn.Module):
    """
    Model incorporating intro + incpetion blocks + outro with final layer
    """

    def __init__(self) -> None:
        super().__init__()
        # assumes in put shape [N, 1, 32, 96]

        self.first = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding="same",
                padding_mode="replicate",
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                padding="same",
                padding_mode="replicate",
            ),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.MaxPool2d(kernel_size=3, stride=1),

            InceptionA(in_channels=16, concat_channels=96),
            InceptionB(in_channels=112, concat_channels=128),
            InceptionB(in_channels=240, concat_channels=128),

            nn.Conv2d(in_channels=368, out_channels=32, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1)
        )

        self.second = nn.Sequential(
            nn.Sigmoid(), 
            nn.Linear(32, n_classes))


    def forward(self, x):
        z = self.first(x)
        z = z.mean(dim=(3, 2))
        z = self.second(z)

        return z


learning_rate = 2e-3
weight_decay = 1e-3
n_epochs = 20


model = Model().to(device)

# Print amount of parameters
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


model.train()

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=0, verbose=True
)
loss_fn = nn.CrossEntropyLoss()

acc_running = 0


config = {
    "architecture": "inceptiion_model",
    "n_epochs": 20,
    "optimizer": "Adam",
    "loss_fn": "CrossEntropyLoss",
    "model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
}

wandb.init(
    project="OTICON",
    entity="metrics_logger",
    settings=wandb.Settings(start_method="thread"),
    config=config,
)

wandb.log({"lr": optimizer.param_groups[0]["lr"]})


for epoch in (bar := trange(n_epochs)):
    # print(epoch)

    for x, y in tqdm(loader_train, leave=False):
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        # print(y_pred.shape)
        # print(y.shape)

        loss = loss_fn(y_pred, y)

        acc = (y_pred.argmax(dim=1) == y).sum() / y.size(0)
        f1 = f1_score(y, y_pred.argmax(dim=1), average='macro')
        # Alpha=0.05 update of running accuracy
        acc_running += 0.05 * (acc.item() - acc_running)
        bar.set_postfix(acc=f"{acc_running:.2f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log with wandb
        wandb.log({"acc_train": acc_running, "loss_train": loss, "f1_train": f1})

    scheduler.step(acc_running)
    wandb.log({"lr": optimizer.param_groups[0]["lr"]})


print(acc_running)


model.eval()

acc_running = 0
y_pred_s = []

for i, (x, y) in (bar := tqdm(enumerate(loader_val, 1), leave=False)):
    x = x.to(device)
    y = y.to(device)

    y_pred = model(x)
    acc = (y_pred.argmax(dim=1) == y).sum() / y.size(0)

    # Mean of accuracy
    acc_running += 1 / i * (acc.item() - acc_running)
    bar.set_postfix(acc=f"{acc_running:.2f}")
    y_pred_s.append(y_pred.argmax(dim=1).detach().cpu().numpy())

# log with wandb
wandb.log({"acc_val": acc_running})

print(acc_running)

# log data, labels, and predictions
wandb.log({"spectrograms": [wandb.Image(im) for im in X_train]})
wandb.log({"predictions": y_pred_s})

