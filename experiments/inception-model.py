# %%
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torch.utils.data import TensorDataset, DataLoader
t.set_float32_matmul_precision("medium")

import torchmetrics as tm
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

import os
import sys

# Add current working directory to path
sys.path.insert(0, os.getcwd())

from data.loader import load_training, load_validation, n_freq, n_time, n_classes
from inception_modules import InceptionA, InceptionB, MeanModule



# %%
# Load data
X_train, y_train = load_training()
X_val, y_val = load_validation()

# To tensors
X_map = lambda X: t.from_numpy(X).to(dtype=t.float)[:, None, :, :]
y_map = lambda y: t.from_numpy(y).to(dtype=t.uint8)
X_train, X_val = map(X_map, (X_train, X_val))
y_train, y_val = map(y_map, (y_train, y_val))

# Statistics
X_train_mean = X_train.mean()
X_train_std = X_train.std()
n_train, n_val = len(X_train), len(X_val)

# Standardize
X_std_map = lambda X: (X - X_train_mean) / X_train_std
X_train, X_val = map(X_std_map, (X_train, X_val))

# To dataset
data_train, data_val = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)

# To data loader
loader_map = lambda data: DataLoader(
    dataset=data,
    batch_size=128,
    num_workers=4,
)
loader_train, loader_val = map(loader_map, (data_train, data_val))



# %%
class InceptionModel(pl.LightningModule):
    def __init__(self, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


        self.accuracy = tm.Accuracy(task="multiclass", num_classes=n_classes)
        self.f1 = tm.F1Score(task="multiclass", num_classes=n_classes, average="macro")


        self.model = nn.Sequential(
            # Input shape [N, 1, 32, 96]
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
            # Global average pool
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            MeanModule(dim=(3, 2)),

            nn.Sigmoid(),
            nn.Linear(32, n_classes)
        )


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y_true)


        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(y_pred, y_true))
        self.log("train_f1", self.f1(y_pred, y_true))

        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y_true)


        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy(y_pred, y_true))
        self.log("val_f1", self.f1(y_pred, y_true))


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_f1"
        }



model = InceptionModel(
    learning_rate=2e-3,
    weight_decay=1e-3
)


WANDB__SERVICE_WAIT = 300
wandb_logger = WandbLogger(
    project="OTICON", 
    entity="metrics_logger"
)

trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=40,
    min_epochs=20,
    logger=wandb_logger,
    precision="16-mixed", #or "32-true"
    callbacks=[
        LearningRateMonitor(logging_interval="epoch")
    ],
)

trainer.fit(model, loader_train, loader_val)