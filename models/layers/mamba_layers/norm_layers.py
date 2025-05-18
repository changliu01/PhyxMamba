import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8, elementwise_affine=True):
        super(RMSNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        # 确定规范化的维度
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        # 计算均方根 (RMS)
        rms = torch.sqrt(torch.mean(x.pow(2), dim=dims, keepdim=True) + self.eps)
        
        # 归一化输入
        x_norm = x / rms
        
        if self.elementwise_affine:
            # 通过广播机制应用缩放参数
            x_norm = x_norm * self.weight
        return x_norm