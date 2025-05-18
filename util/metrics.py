import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics_utils import smape, vpt_smape, vpt_rmse
from dysts.metrics import estimate_kl_divergence, average_hellinger_distance
from dysts.analysis import gp_dim
from tqdm import tqdm
from torchsummary import summary
from thop import profile
from prettytable import PrettyTable

def relative_l2_error(x, x_hat,steps=None):

    '''
    x: np.ndarray, shape (batch, seq_len, input_dim)
    x_hat: np.ndarray, shape (batch, seq_len, input_dim)
    '''
    if steps is not None:
        x = x[:, :steps]
        x_hat = x_hat[:, :steps]

    numerator = np.linalg.norm(x-x_hat, ord=2, axis=1)
    denominator = np.linalg.norm(x, ord=2, axis=1)

    return np.mean(numerator/denominator)

def SMAPE(y_true, y_pred, steps=None):
    
    batch_size = len(y_true)
    smapes = []
    for i in range(batch_size):
        if steps is None:
            smapes.append(smape(y_true[i], y_pred[i]))
        else:
            smapes.append(smape(y_true[i][:steps], y_pred[i][:steps]))
    
    return np.mean(smapes)

def VPT(y_true, y_pred, threshold=30):
    
    batch_size = len(y_true)
    vpts = []
    
    for i in range(batch_size):
        vpt_value = vpt_smape(y_true[i], y_pred[i], threshold)
        vpts.append(vpt_value)

    return np.mean(vpts)

def VPS(y_true, y_pred, threshold=0.01):
    batch_size = len(y_true)
    vps = []
    for i in range(batch_size):
        vpt_value = vpt_rmse(y_true[i], y_pred[i], threshold)
        vps.append(vpt_value)
    return np.mean(vps)


def MAE(y_true, y_pred, mask_pred=None, mask_true=None):

    if mask_pred is not None:
        abs_error = np.abs(y_true - y_pred) * mask_pred
        total_error = np.sum(abs_error)
        total_count = np.sum(mask_pred)
        if total_count == 0:
            return 0
        return total_error / total_count
    else:
        return np.mean(np.abs(y_true - y_pred))

def MSE(y_true, y_pred, mask_pred=None, mask_true=None):

    if mask_pred is not None:
        squared_error = np.square(y_true - y_pred) * mask_pred
        total_error = np.sum(squared_error)
        total_count = np.sum(mask_pred)
        if total_count == 0:
            return 0
        return total_error / total_count
    else:
        return np.mean(np.square(y_true - y_pred))

def KLD(y_true, y_pred, n_samples=1000, sigma_scale=1.0, upper_bound=np.inf):

    '''
    y_true: torch.Tensor, shape (batch_size, seq_len, input_dim)
    y_pred: torch.Tensor, shape (batch_size, seq_len, input_dim)
    '''

    batch_size = y_true.shape[0]
    klds = []
    for i in tqdm(range(batch_size)):
        kl = estimate_kl_divergence(y_true[i], y_pred[i], n_samples, sigma_scale)
        if kl is not None and (not np.isnan(kl)) and (not np.isinf(kl)) and (kl < upper_bound):
            klds.append(kl)
    return np.mean(klds)

def gpdim_rmse(y_true, y_pred):

    '''
    y_true: np.ndarray, shape (batch_size, seq_len, input_dim)
    y_pred: np.ndarray, shape (batch_size, seq_len, input_dim)
    '''
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 3

    batch_size = y_true.shape[0]
    gpdim_true = []
    gpdim_pred = []

    for i in tqdm(range(batch_size)):
        try:
            gpdim_true.append(gp_dim(y_true[i]))
            gpdim_pred.append(gp_dim(y_pred[i]))
        except:
            continue
    return np.sqrt(np.mean(np.square(np.array(gpdim_true) - np.array(gpdim_pred))))

def Dh(y_true, y_pred):

    '''
    y_true: np.ndarray, shape (batch_size, seq_len, input_dim)
    y_pred: np.ndarray, shape (batch_size, seq_len, input_dim)
    '''

    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 3

    batch_size = y_true.shape[0]
    dhs = []
    for i in tqdm(range(batch_size)):
        try:
            dh = average_hellinger_distance(y_true[i], y_pred[i])
            dhs.append(dh)
        except:
            continue
    return np.mean(dhs)

def R2(y_true, y_pred, mask_pred=None, mask_true=None):

    if mask_pred is not None:

        mask_y_true = y_true[mask_true == 1]
        mask_y_pred = y_pred[mask_pred == 1]

        if len(mask_y_true) == 0:
            return np.nan
        y_mean = np.mean(mask_y_true)

        ss_res = np.sum(np.square(mask_y_true - mask_y_pred))
        ss_tot = np.sum(np.square(mask_y_true - y_mean))

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - ss_res / ss_tot
    
    else:
        return 1 - np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true)))


# 记录推理内存开销、显存开销、计算开销FLOPs、参数量
def get_flops(args, model):

    try:
        device = next(model.parameters()).device
        print(device)
    except StopIteration:
        # 这可能意味着模型没有参数，例如一个纯粹的 nn.Sequential 且为空，
        # 或者是一个没有注册任何参数或缓冲区的自定义模块。
        # 在这种情况下，您可能需要根据您的模型结构具体处理，
        # 或者在 __init__ 中显式存储一个 device 属性。
        print("模型没有参数。")
        # 可以设置一个默认设备，或者根据 args 来决定
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 示例
        # 或者如果您在args中定义了device:
        # device = args.device
        pass # 根据您的逻辑处理

    dummy_input = torch.randn(1, args.lookback, args.variable_num).to(device)
    
    try:
        summary(model.cuda(), (args.lookback, args.variable_num))
    except:
        print('Summary Error')
    flops, params = profile(model, (dummy_input,), verbose=False)
    # print(f'FLOPs: {flops}, Params: {params}')

    return flops, params

def get_inference_times(args, model):

    try:
        device = next(model.parameters()).device
    except StopIteration:
        print("模型没有参数。")
        pass 

    dummy_input = torch.randn(1, args.lookback, args.variable_num).to(device)

    for _ in tqdm(range(50), desc='warmup'):
        model(dummy_input)
    
    # 记录推理时间开销
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    mean_times = 100
    if args.system == 'EEG':
        pred_length = 600
    else:
        pred_length = 300
    iterations = pred_length // args.token_size

    # 测速
    times = torch.zeros(mean_times) # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in tqdm(range(mean_times), desc='inference time'):
            starter.record()
            for _ in range(iterations):
                model(dummy_input)
            ender.record()
            torch.cuda.synchronize() # 同步GPU时间
            curr_time = starter.elapsed_time(ender) # 计算时间
            times[iter] = curr_time
    mean_time = times.mean().item()
    std_time = times.std().item()
    return mean_time, std_time, 1000/mean_time

def get_inference_times_darts(args, model):

    try:
        device = next(model.parameters()).device
    except StopIteration:
        print("模型没有参数。")
        pass 

    dummy_input = torch.randn(args.lookback, args.variable_num).to(device)

    for _ in tqdm(range(50), desc='warmup'):
        model(dummy_input)
    
    # 记录推理时间开销
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    mean_times = 100
    if args.system == 'EEG':
        pred_length = 600
    else:
        pred_length = 300
    iterations = pred_length // args.token_size

    # 测速
    times = torch.zeros(mean_times) # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in tqdm(range(mean_times), desc='inference time'):
            starter.record()
            for _ in range(iterations):
                model(dummy_input)
            ender.record()
            torch.cuda.synchronize() # 同步GPU时间
            curr_time = starter.elapsed_time(ender) # 计算时间
            times[iter] = curr_time
    mean_time = times.mean().item()
    std_time = times.std().item()
    return mean_time, std_time, 1000/mean_time



def get_model_size(model, train_logger):

    trainable_params = 0
    non_trainable_params = 0
    trainable_size = 0
    non_trainable_size = 0

    params_wo_mtp = 0
    size_wo_mtp = 0
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())

    for name, param in model.named_parameters():
        num_elements = param.numel()
        size_bytes = num_elements * param.element_size()
        
        if 'mtp' not in name:
            params_wo_mtp += num_elements
            size_wo_mtp += size_bytes

        if param.requires_grad:
            trainable_params += num_elements
            trainable_size += size_bytes
        else:
            non_trainable_params += num_elements
            non_trainable_size += size_bytes

    # 转换为 K（千）单位，保留两位小数
    trainable_params_k = trainable_params / (1024 * 1024)
    non_trainable_params_k = non_trainable_params / (1024 * 1024)
    total_params_k = total_params / (1024 * 1024)  # 直接用 total_params 转换
    params_wo_mtp_k = params_wo_mtp / (1024 * 1024)

    # 转换为 MB，保留两位小数
    trainable_size_mb = trainable_size / (1024 * 1024)
    non_trainable_size_mb = non_trainable_size / (1024 * 1024)
    total_size_mb = total_size / (1024 * 1024)
    size_wo_mtp_mb = size_wo_mtp / (1024 * 1024)
    # 创建 PrettyTable 并设置列名
    param_table = PrettyTable()
    param_table.field_names = ["Trainable", "Non-Trainable", "Total", "Total (w/o MTP)", "Model Size", "Model Size (w/o MTP)"]

    # 添加一行，显示参数量 (K) 和内存大小 (MB)
    param_table.add_row([
        f"{trainable_params_k:7f} M",
        f"{non_trainable_params_k:.7f} M",
        f"{total_params_k:.7f} M",
        f"{params_wo_mtp_k:.7f} M",
        f"{total_size_mb:.7f} MB",
        f"{size_wo_mtp_mb:.7f} MB"
    ])

    # 打印表格
    print(param_table)
    train_logger.info(param_table)