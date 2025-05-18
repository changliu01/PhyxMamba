import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

# 计算预测误差的函数
def compute_forecast_errors(model, data_loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            error = torch.mean((outputs - targets) ** 2, dim=1)  # 均方误差
            errors.extend(error.cpu().numpy())
    model.train()
    return np.array(errors)

