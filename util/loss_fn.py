import torch
import torch.nn as nn

def rational_quadratic_kernel(x, y, sigma_list=[0.2, 0.5, 0.9, 1.3]):
    if x.dim() == 3:
        x = x.squeeze(1)  # [B, N]
    if y.dim() == 3:
        y = y.squeeze(1)  # [B, N]
    x = x.unsqueeze(1)  # [B, 1, N]
    y = y.unsqueeze(0)  # [1, B, N]
    
    squared_dist = torch.sum((x - y) ** 2, dim=-1)  # [B, B]
    
    # 将 sigma_list 转换为 tensor 并使用广播机制
    sigma = torch.tensor(sigma_list).view(-1, 1, 1).to(x.device)  # [len(sigma_list), 1, 1]
    sigma_squared = sigma ** 2  # [len(sigma_list), 1, 1]
    
    # 计算 kernel 值
    kernel_val = sigma_squared / (sigma_squared + squared_dist)  # [len(sigma_list), B, B]
    
    # 对所有的 sigma_list 求和
    kernel_val = kernel_val.sum(dim=0)  # [B, B]
    
    return kernel_val

# MMD 计算（不变）
def compute_mmd(x, y, mean_value, variance_value):
    if x.dim() == 3:
        x = x.squeeze(1)  # [B, N]
    if y.dim() == 3:
        y = y.squeeze(1)  # [B, N]
    if mean_value.dim() == 1:
        mean_value = mean_value.unsqueeze(0)  # [1, N]
    if variance_value.dim() == 1:
        variance_value = variance_value.unsqueeze(0)  # [1, N]
    x = (x - mean_value) / torch.sqrt(variance_value + 1e-6)  # [B, N]
    y = (y - mean_value) / torch.sqrt(variance_value + 1e-6)  # [B, N]
    B = x.size(0)
    xx = rational_quadratic_kernel(x, x)
    yy = rational_quadratic_kernel(y, y)
    xy = rational_quadratic_kernel(x, y)
    term1 = (xx.sum()) / (B * B)
    term2 = (yy.sum()) / (B * B)
    term3 = xy.sum() / (B * B)
    return (term1 + term2 - 2 * term3).clamp(min=0)

# 多步无条件 MMD
def unconditional_mmd_multi_step(input_traj, pred_traj, mean, variance, steps=None):
    """
    计算多步无条件 MMD: 平均 D(mu*, (S_theta^t)_# mu*) for t in steps
    input_traj: 输入轨迹 [B, T, N]
    pred_traj: 模型预测轨迹 [B, H, N]
    mean: 均值 [N]
    variance: 方差 [N]
    steps: 使用的预测时间步列表，默认为所有步 [0, 1, ..., H-1]
    返回: 平均 MMD 值
    """
    B, T, N = input_traj.shape
    H = pred_traj.shape[1]
    
    # 初始状态作为 mu* 的样本
    true_samples = input_traj[:, 0, :].unsqueeze(1)  # [B, 1, N]
    
    # 默认使用所有预测步
    if steps is None:
        steps = range(H)  # [0, 1, 2, ..., H-1]
    
    mmd_sum = 0.0
    for t in steps:
        model_samples = pred_traj[:, t, :].unsqueeze(1)  # [B, 1, N]
        mmd_sum += compute_mmd(true_samples, model_samples, mean, variance)
    
    return mmd_sum / len(steps)

# 多步条件 MMD
def conditional_mmd_multi_step(input_traj, true_traj, pred_traj, mean, variance, steps=None):
    """
    计算多步条件 MMD: 平均 D((S^t)_# mu*, (S_theta^t)_# mu*) for t in steps
    input_traj: 输入轨迹 [B, T, N]
    true_traj: 真实未来轨迹 [B, H, N]
    pred_traj: 模型预测轨迹 [B, H, N]
    mean: 均值 [N]
    variance: 方差 [N]
    steps: 使用的预测时间步列表，默认为所有步 [0, 1, ..., H-1]
    返回: 平均 MMD 值
    """
    # B, T, N = input_traj.shape
    H = pred_traj.shape[1]
    
    # 默认使用所有预测步
    if steps is None:
        steps = range(H)  # [0, 1, 2, ..., H-1]
    
    mmd_sum = 0.0
    for t in steps:
        true_evolved = true_traj[:, t, :].unsqueeze(1)  # [B, 1, N]
        model_evolved = pred_traj[:, t, :].unsqueeze(1)  # [B, 1, N]
        mmd_sum += compute_mmd(true_evolved, model_evolved, mean, variance)
    
    return mmd_sum / len(steps)


class MMDLoss(nn.Module):

    def __init__(self, args):
        
        super().__init__()

        self.args = args

    def forward(self, input_traj, pred_traj, label_traj, mean, variance, steps=None):
        """
        input_traj: 输入轨迹 [B, T, N]
        pred_traj: 模型预测轨迹 [B, H, N]
        label_traj: 真实未来轨迹 [B, H, N]
        mean: 均值 [N]
        variance: 方差 [N]
        steps: 使用的预测时间步列表，默认为所有步 [0, 1, ..., H-1]
        返回: MMD 损失值
        """

        uncond_loss = unconditional_mmd_multi_step(input_traj, pred_traj, mean, variance, steps)
        cond_loss = conditional_mmd_multi_step(input_traj, label_traj, pred_traj, mean, variance, steps) 

        mmd_loss = uncond_loss * self.args.lambda_uncond_mmd + cond_loss * self.args.lambda_cond_mmd

        
        return mmd_loss