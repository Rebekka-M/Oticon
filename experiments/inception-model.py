#%%
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import os
import sys

sys.path.insert(0, os.getcwd())

from data.loader import load_training, load_validation, n_freq, n_time, n_classes
from inception import InceptionA, InceptionB

#%%

force_cpu = False

if t.cuda.is_available() and not force_cpu:
    device = t.device("cuda")
else:
    device = t.device("cpu")


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

#%%


class Model(nn.Module):
    """
    Model incorporating intro + incpetion blocks + outro with final layer
    """
    def __init__(self) -> None:
        super().__init__()
        # assumes in put shape [N, 1, 32, 96]

        #before inception
        self.conv1 = nn.Conv2d(in_channels = 1,  out_channels=16, kernel_size=3, padding="same", padding_mode="replicate") #TO decide
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels=16, kernel_size=3, padding="same", padding_mode="replicate") #To decide
        self.pad1 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1)

        # inception
        self.inceptionA = InceptionA(in_channels=16)
        self.inceptionB1 = InceptionB(in_channels=112)
        self.inceptionB2 = InceptionB(in_channels=240)

        # after inception
        self.conv3 = nn.Conv2d(in_channels = 368, out_channels= 32,  kernel_size=1)
        self.average_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.global_average_pool = nn.AdaptiveAvgPool2d(output_size=32) 

        self.sigmoid_class = nn.Sigmoid()
        self.fc_flatten = nn.Flatten()
        self.fc_final = nn.Linear(32768, n_classes)

    def forward(self, x):

        z = self.conv1(x)
        z = self.conv2(z)
        z = self.pad1(z)
        z = self.maxpool1(z)
        
        z = self.inceptionA(z)
        z = self.inceptionB1(z)
        z = self.inceptionB2(z)

        z = self.conv3(z)
        z = self.average_pool(z)
        
        z = self.global_average_pool(z)
        z = self.sigmoid_class(z)
        z = self.fc_flatten(z)
        z = self.fc_final(z)
       

        return z


learning_rate = 2e-3
weight_decay = 1e-3
n_epochs = 20

model = Model().to(device)

# Print amount of parameters
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


model.train()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=0, verbose=True)
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate*10, steps_per_epoch=n_train//128, epochs=n_epochs, anneal_strategy="cos")
loss_fn = nn.CrossEntropyLoss()

acc_running = 0

for epoch in (bar:=trange(n_epochs)):
    # print(epoch)

    for x, y in tqdm(loader_train, leave=False):
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        # print(y_pred.shape)
        # print(y.shape)
        
        loss = loss_fn(y_pred, y)
        

        acc = (y_pred.argmax(dim=1) == y).sum() / y.size(0)
        # Alpha=0.05 update of running accuracy
        acc_running += 0.05 * (acc.item() - acc_running)
        bar.set_postfix(acc=f"{acc_running:.2f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step(acc_running)
    # scheduler.step()


print(acc_running)



model.eval()

acc_running = 0

for i, (x, y) in (bar:=tqdm(enumerate(loader_val, 1), leave=False)):
    x = x.to(device)
    y = y.to(device)

    y_pred = model(x)
    acc = (y_pred.argmax(dim=1) == y).sum() / y.size(0)

    # Mean of accuracy
    acc_running += 1/i * (acc.item() - acc_running)
    bar.set_postfix(acc=f"{acc_running:.2f}")


print(acc_running)


if __name__ == "__main__":
    pass
# %%
