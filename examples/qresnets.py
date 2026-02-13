# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: dq
#     language: python
#     name: python3
# ---

# %%
import random

import deepquantum as dq
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

random.seed(43)
np.random.seed(43)
torch.manual_seed(43)


# %%
class Y1Dataset(Dataset):
    def __init__(self, omega, a):
        """
        Args:
            omega (tensor): A tensor of omega values.
            a (tensor): A complex number representing the amplitude.
        """
        self.x_values = torch.linspace(-6, 6, 600).view(-1, 1)  # 2D tensor with shape (num_samples, 1)
        self.y_values = self.y1(self.x_values, omega, a)

    def y1(self, x, omega, a):
        sum_result = torch.zeros_like(x, dtype=torch.complex64)
        for w in omega:
            sum_result += a * torch.exp(1j * w * x) + torch.conj(a) * torch.exp(-1j * w * x)
        return sum_result.real

    def __len__(self):
        return len(self.x_values)

    def __getitem__(self, idx):
        return self.x_values[idx], self.y_values[idx]


class Y2Dataset(Dataset):
    def __init__(self, omega, amps):
        """
        Args:
            omega (tensor): A tensor of omega values.
            amps (tensor): A tensor of complex numbers representing the amplitudes.
        """
        self.x_values = torch.linspace(-6, 6, 600).view(-1, 1)  # 2D tensor with shape (num_samples, 1)
        self.y_values = self.y2(self.x_values, omega, amps)

    def y2(self, x, omega, amps):
        sum_result = torch.zeros_like(x, dtype=torch.complex64)
        for w, a in zip(omega, amps, strict=True):
            sum_result += a * torch.exp(1j * w * x) + torch.conj(a) * torch.exp(-1j * w * x)
        return sum_result.real

    def __len__(self):
        return len(self.x_values)

    def __getitem__(self, idx):
        return self.x_values[idx], self.y_values[idx]


# %%
# Example usage:
omega1 = torch.tensor([0.0, 1.0, -1.0])
a = torch.tensor(0.1)
dataset_y1_omega1 = Y1Dataset(omega1, a)


# Plot
plt.figure(figsize=(10, 6))
plt.plot(
    dataset_y1_omega1.x_values.flatten().numpy(),
    dataset_y1_omega1.y_values.flatten().numpy(),
    'k--',
    label='y1(x) with Ω1',
)
plt.xlabel('x')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Example usage:
omega2 = torch.tensor([0.0, 1.0, 0.5])
a = torch.tensor(0.1j)
dataset_y1_omega2 = Y1Dataset(omega2, a)


# Plot
plt.figure(figsize=(10, 6))
plt.plot(
    dataset_y1_omega2.x_values.flatten().numpy(),
    dataset_y1_omega2.y_values.flatten().numpy(),
    'k--',
    label='y1(x) with Ω2',
)
plt.xlabel('x')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Example usage:
omega2 = torch.tensor([0.0, 1.0, 0.5])
amps = torch.tensor([0.1j, 0.15, 0.12])  # 并非所有的系数都能拟合，取决于变分线路
dataset_y2_omega2 = Y2Dataset(omega2, amps)


# Plot
plt.figure(figsize=(10, 6))
plt.plot(
    dataset_y2_omega2.x_values.flatten().numpy(),
    dataset_y2_omega2.y_values.flatten().numpy(),
    'k--',
    label='y2(x) with Ω2',
)
plt.xlabel('x')
plt.legend()
plt.grid(True)
plt.show()


# %%
def get_loader(dataset, split_ratio=0.8, batch_size=8):
    # Splitting the dataset into training and validation datasets
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # You can now use DataLoader to load these datasets if needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# %%
get_loader(dataset_y1_omega1)


# %%
class QNN(nn.Module):
    def __init__(self, residual=False):
        super().__init__()
        self.residual = residual
        if residual:
            self.qnn = dq.QubitCircuit(2)
            self.qnn.u3(wires=1)
            self.qnn.ry(wires=0)
            self.qnn.ry(wires=1, encode=True, controls=0)
            self.qnn.ry(wires=0)
            self.qnn.u3(wires=1)
            self.qnn.observable(wires=1, basis='z')
            self.qnn.observable(wires=[0, 1], basis='zz')
        else:
            self.qnn = dq.QubitCircuit(1)
            self.qnn.u3(wires=0)
            self.qnn.ry(wires=0, encode=True)
            self.qnn.u3(wires=0)
            self.qnn.observable(wires=0, basis='z')

    def forward(self, x):
        self.qnn(data=x)
        exp = self.qnn.expectation()
        if self.residual:
            exp = (exp[:, [0]] + exp[:, [1]]) / 2
        return exp


# %%
def train_model(epochs, model, dataset):
    train_loader, val_loader = get_loader(dataset)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        # Training phase
        model.train()  # Set the model to training mode
        for x, y in train_loader:
            yhat = model(x)
            loss = torch.nn.functional.mse_loss(yhat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch == 0:
                print(loss.item())

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation
            for x, y in val_loader:
                yhat = model(x)
                loss = torch.nn.functional.mse_loss(yhat, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print('Validation - Epoch', epoch, 'Average Valid Loss:', avg_val_loss)

    return model


# %%
def plot_predictions(dataset, label_dataset, trained_model, label_model):
    if not isinstance(trained_model, list):
        # Generate predictions using the model
        trained_model.eval()
        with torch.no_grad():
            predictions = trained_model(dataset.x_values).flatten().numpy()

        # Plot the original y(x) and the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(
            dataset.x_values.flatten().numpy(), dataset.y_values.flatten().numpy(), 'k--', label=label_dataset, zorder=2
        )
        plt.plot(dataset.x_values.flatten().numpy(), predictions, 'gray', linewidth=6, label=label_model, zorder=1)
    else:
        # Plot the original y(x) and the predictions
        plt.figure(figsize=(10, 6))
        plt.plot(
            dataset.x_values.flatten().numpy(), dataset.y_values.flatten().numpy(), 'k--', label=label_dataset, zorder=2
        )
        for model, label in zip(trained_model, label_model, strict=True):
            model.eval()
            with torch.no_grad():
                predictions = model(dataset.x_values).flatten().numpy()
            if 'residual' in label:
                color = 'red'
            elif 'traditional' in label:
                color = 'gray'
            plt.plot(dataset.x_values.flatten().numpy(), predictions, color, linewidth=6, label=label, zorder=1)

    plt.xlabel('x')
    plt.legend()
    plt.grid(True)
    plt.show()


# %% [markdown]
# # 实验1（dataset_y1_omega1）

# %%
qnn_traditional = QNN(residual=False)
qnn_traditional.qnn.draw()

# %%
trained_qnn_traditional = train_model(epochs=30, model=qnn_traditional, dataset=dataset_y1_omega1)

# %%
qnn_residual = QNN(residual=True)
qnn_residual.qnn.draw()

# %%
trained_qnn_residual = train_model(epochs=30, model=qnn_residual, dataset=dataset_y1_omega1)

# %%
plot_predictions(
    dataset_y1_omega1,
    label_dataset='y1(x) with Ω1',
    trained_model=[trained_qnn_traditional, trained_qnn_residual],
    label_model=['qnn_traditional', 'qnn_residual'],
)

# %% [markdown]
# # 实验2（dataset_y1_omega2）

# %%
qnn_traditional = QNN(residual=False)
qnn_traditional.qnn.draw()

# %%
trained_qnn_traditional = train_model(epochs=30, model=qnn_traditional, dataset=dataset_y1_omega2)

# %%
qnn_residual = QNN(residual=True)  # 参数初始化敏感，需要多实例化几次
qnn_residual.qnn.draw()

# %%
trained_qnn_residual = train_model(epochs=30, model=qnn_residual, dataset=dataset_y1_omega2)

# %%
plot_predictions(
    dataset_y1_omega2,
    label_dataset='y1(x) with Ω2',
    trained_model=[trained_qnn_traditional, trained_qnn_residual],
    label_model=['qnn_traditional', 'qnn_residual'],
)

# %% [markdown]
# # 实验3（dataset_y2_omega2）

# %% [markdown]
#

# %%
qnn_traditional = QNN(residual=False)
qnn_traditional.qnn.draw()

# %%
trained_qnn_traditional = train_model(epochs=30, model=qnn_traditional, dataset=dataset_y2_omega2)

# %%
qnn_residual = QNN(residual=True)  # 参数初始化敏感，需要多实例化几次
qnn_residual.qnn.draw()

# %%
trained_qnn_residual = train_model(epochs=30, model=qnn_residual, dataset=dataset_y2_omega2)

# %%
plot_predictions(
    dataset_y2_omega2,
    label_dataset='y2(x) with Ω2',
    trained_model=[trained_qnn_traditional, trained_qnn_residual],
    label_model=['qnn_traditional', 'qnn_residual'],
)

# %%

# %%

# %%

# %%
