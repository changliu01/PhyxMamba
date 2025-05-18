import os
import torch
import random
import subprocess
import numpy as np
import fcntl
import time
from typing import List

import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
# from .metrics_old import SMAPE, MAE, MSE, R2, correlation_dimension, kl_divergence, VPT, energy_spectrum, get_dstsp, get_pse
from .metrics import SMAPE, VPT, MAE, MSE, gpdim_rmse, Dh, KLD, R2, VPS, relative_l2_error, get_flops,  get_inference_times
import fcntl

def set_cpu_num(cpu_num):
    if cpu_num <= 0: return
    
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AutoGPU():
    
    def __init__(self, memory_size, allow_gpu_list):
        
        self.memory_size = memory_size
        
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")

        self.free_memory = []
        for i, free_memory_str in enumerate(output):
            self.free_memory.append(int(free_memory_str))

        if len(allow_gpu_list) == 0:
            self.allow_gpu_list = list(range(len(self.free_memory)))
        else:
            self.allow_gpu_list = allow_gpu_list

    def update_free_memory(self):
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")
        self.free_memory = [int(free_memory_str) for free_memory_str in output]
    
    def get_required_memory(self, model_class, model_args, data):

        estimated_memory = 2000  # Start with an initial estimate (in MB)
        step = 500  # Increase step size (in MB)

        while True:
            try:
                # Try to load the model with the current estimated memory
                model = model_class(*model_args).cuda()
                dummy_input = torch.randn(*data).cuda()
                output = model(dummy_input)
                del model, dummy_input, output
                torch.cuda.empty_cache()
                break  # If successful, break the loop
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    estimated_memory += step
                else:
                    raise e  # Raise any other exceptions
        
        return estimated_memory
     
    def choice_gpu(self):

        flag = False
        for i, free_memory in enumerate(self.free_memory):

            if free_memory >= self.memory_size and (i in self.allow_gpu_list):
                
                flag = True
                self.free_memory[i] -= self.memory_size
                print(f"GPU-{i}: {free_memory}MB -> {self.free_memory[i]}MB")
                
                return i
        
        if not flag:
            print(f"SubProcess[{os.getpid()}]: No GPU can use, switch to CPU!")
            return -1
        
def validation_metrics(true, pred, epoch, max_epoch, mask_pred=None, mask_true=None, origin_pred=None, origin_true=None):

    if mask_pred is not None:
        smape, mae, mse, r2 = SMAPE(origin_true, origin_pred, list_y=True), MAE(true, pred, mask_pred, mask_true), MSE(true, pred, mask_pred, mask_true), R2(true, pred, mask_pred, mask_true)
    else:
        smape, mae, mse, r2 = SMAPE(true, pred), MAE(true, pred), MSE(true, pred), R2(true, pred)
    print(f'Validation @ Epoch[{epoch}/{max_epoch}]')
    print('----------------------------------------------------------------')
    print(f'SMAPE: {smape:.5f},  MAE: {mae:.5f}, MSE: {mse:.5f}, R2: {r2:.5f}')

# def test_metrics(true, pred, mask_pred=None, mask_true=None, origin_pred=None, origin_true=None):
#     if mask_pred is not None:
#         smape, cdim, mae, mse, r2, kld = SMAPE(origin_true, origin_pred, list_y=True), correlation_dimension(origin_true, origin_pred, list_y=True), MAE(true, pred, mask_pred, mask_true), MSE(true, pred, mask_pred, mask_true), R2(true, pred, mask_pred, mask_true), kl_divergence(origin_true, origin_pred, list_y=True)
#         vpt, horizons = VPT(origin_true, origin_pred)
#     else:
#         smape, cdim, mae, mse, r2, kld = SMAPE(true, pred), correlation_dimension(true, pred), MAE(true, pred), MSE(true, pred), R2(true, pred), kl_divergence(true, pred)
#         vpt, horizons = VPT(true, pred)
#     return smape, cdim, mae, mse, r2, kld, vpt, horizons


def test_metrics_long(true, pred, args):

    smape_dict = {}
    _, T, __ = true.shape
    eval_steps = [int(T*0.1), int(T * 0.2), int(T * 0.4), int(T * 0.5), int(T * 0.6), int(T * 0.8), T]
    for steps in eval_steps:
        smape_dict[steps] = SMAPE(true, pred, steps)
    vpt = VPT(true, pred)

    if (args.data_type != 'real') and (args.data_type != 'EEG') and (args.data_type != 'ECG'):
        cd_rmse = gpdim_rmse(true, pred)
        dh = Dh(true, pred)
        if args.model == 'ESN':
            kld = KLD(true, pred, upper_bound=args.kld_upper_bound)
        else:
            kld = KLD(true, pred)
    else:
        cd_rmse, dh, kld = 0, 0, 0

    return smape_dict, vpt, cd_rmse, dh, kld
    
def test_metrics_short(true, pred, short_steps):

    mae = MAE(true[:, :short_steps, :], pred[:, :short_steps, :])
    mse = MSE(true[:, :short_steps, :], pred[:, :short_steps, :])
    r2 = R2(true[:, :short_steps, :], pred[:, :short_steps, :])
    
    return mae, mse, r2

def test_metrics_complexity(args, model):
    flops, params = get_flops(args, model)
    inference_time, fps = get_inference_times(args, model)

    return flops, params, inference_time, fps


def test_metrics_bench(true, pred):

    smape = SMAPE(true, pred)
    mae = MAE(true, pred)
    mse = MSE(true, pred)
    r2 = R2(true, pred)
    
    return smape, mae, mse, r2

def print_test_metrics(args, smape_dict, vpt, cd_rmse, mae, mse, dh, kld):

    output_dir_base = args.print_metrics_dir
    exp_type = args.exp_type
    output_dir = os.path.join(output_dir_base, args.system)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{exp_type}')
    
    if args.exp_type == 'normal':
        indicator_title = ['normal', 'vpt', 'smape0.1', 'smape0.4', 'mae', 'mse', 'cdrmse', 'kld', 'dh', 'smape0.2', 'smape0.5', 'smape0.6', 'smape0.8', 'smape1.0']
        if not os.path.exists(file_path + '.txt'):
            with open(file_path + '.txt', 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.writelines('\t'.join(indicator_title) + '\n')
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
        indicator_list = [args.normal]

    elif args.exp_type == 'noise':
        indicator_title = ['noise_ratio', 'vpt', 'smape0.1', 'smape0.4', 'mae', 'mse', 'cdrmse', 'kld', 'dh', 'smape0.2', 'smape0.5', 'smape0.6', 'smape0.8', 'smape1.0']
        if not os.path.exists(file_path + '.txt'):
            with open(file_path + '.txt', 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.writelines('\t'.join(indicator_title) + '\n')
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
        indicator_list = [args.noise]
    elif args.exp_type == 'ablation':
        indicator_title = ['u_mmd', 'c_mmd', 'emb', 'mtp', 'hier', 'stage2', 'vpt', 'smape0.1', 'smape0.4', 'mae', 'mse', 'cdrmse', 'kld', 'dh', 'smape0.2', 'smape0.5', 'smape0.6', 'smape0.8', 'smape1.0']
        if not os.path.exists(file_path + '.txt'):
            with open(file_path + '.txt', 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.writelines('\t'.join(indicator_title) + '\n')
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
        indicator_list = [args.lambda_uncond_mmd, args.lambda_cond_mmd, args.embedding_method, args.mtp_steps, args.hier_layers, 1 if not args.skip_stage2 else 0]

    elif args.exp_type == 'hyperparameter':

        indicator_title = ['tau', 'delay_emb', 'mtp_steps', 'hlayers', 'u_mmd', 'c_mmd', 'lambda_mtp_loss', 'lookback', 'token_size', 'test_warmup_steps', 'horizon_multistep', 'd_model', 'mamba_d_state', 'mamba_expand', 'mamba_headdim', 'vpt', 'smape0.1', 'smape0.4', 'mae', 'mse', 'cdrmse', 'kld', 'dh', 'smape0.2', 'smape0.5', 'smape0.6', 'smape0.8', 'smape1.0']
        if not os.path.exists(file_path + '.txt'):
            with open(file_path + '.txt', 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.writelines('\t'.join(indicator_title) + '\n')
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
        indicator_list = [args.delay_tau, args.delay_emb_dim, args.mtp_steps, args.hier_layers, args.lambda_uncond_mmd, args.lambda_cond_mmd, args.lambda_mtp_loss, args.lookback, args.token_size, args.test_warmup_steps, args.horizon_multistep, args.d_model, args.mamba_d_state, args.mamba_expand, args.mamba_headdim]

    elif args.exp_type == 'data_ratio':

        indicator_title = ['data_ratio', 'vpt', 'smape0.1', 'smape0.4', 'mae', 'mse', 'cdrmse', 'kld', 'dh', 'smape0.2', 'smape0.5', 'smape0.6', 'smape0.8', 'smape1.0']
        if not os.path.exists(file_path + '.txt'):
            with open(file_path + '.txt', 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.writelines('\t'.join(indicator_title) + '\n')
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
        indicator_list = [args.final_data_ratio]

    elif args.exp_type == 'partial_observe':
        indicator_title = ['partial_observe_dim', 'vpt', 'smape0.1', 'smape0.4', 'mae', 'mse', 'cdrmse', 'kld', 'dh', 'smape0.2', 'smape0.5', 'smape0.6', 'smape0.8', 'smape1.0']
        if not os.path.exists(file_path + '.txt'):
            with open(file_path + '.txt', 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.writelines('\t'.join(indicator_title) + '\n')
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
        indicator_list = [-args.partial_observe_dim]

    else:
        raise NotImplementedError

    file_path = os.path.join(output_dir, f'{exp_type}.txt')
    metrics_list = [vpt, smape_dict[list(smape_dict.keys())[0]], smape_dict[list(smape_dict.keys())[2]], mae, mse, cd_rmse, kld, dh, smape_dict[list(smape_dict.keys())[1]], smape_dict[list(smape_dict.keys())[3]], smape_dict[list(smape_dict.keys())[4]], smape_dict[list(smape_dict.keys())[5]], smape_dict[list(smape_dict.keys())[6]]]

    with open(file_path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        line = '\t'.join(f'{indi}' for indi in indicator_list) + '\t' + '\t'.join(f'{num:.5f}'for num in metrics_list) + '\n'
        f.writelines(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)



def print_test_metrics_baseline(args, smape_dict, vpt, cd_rmse, mae, mse, dh, kld):

    output_dir_base = f'./results/metrics_results'
    output_dir = os.path.join(output_dir_base, args.system, args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.noise > 0:
        file_path = os.path.join(output_dir, 'noise.txt')
        indicator_title = ['noise', 'vpt', 'smape0.1', 'smape0.4', 'mae', 'mse', 'cdrmse', 'kld', 'dh', 'smape0.2', 'smape0.5', 'smape0.6', 'smape0.8', 'smape1.0']
        indicator_list = [args.noise]
    elif args.final_data_ratio < 1:
        file_path = os.path.join(output_dir, 'data_ratio.txt')
        indicator_title = ['data_ratio', 'vpt', 'smape0.1', 'smape0.4', 'mae', 'mse', 'cdrmse', 'kld', 'dh', 'smape0.2', 'smape0.5', 'smape0.6', 'smape0.8', 'smape1.0']
        indicator_list = [args.final_data_ratio]
    else:
        raise NotImplementedError
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.writelines('\t'.join(indicator_title) + '\n')
    # indicator_list = [args.model, args.input_scaling, args.leak]

    metrics_list = [vpt, smape_dict[list(smape_dict.keys())[0]], smape_dict[list(smape_dict.keys())[2]], mae, mse, cd_rmse, kld, dh, smape_dict[list(smape_dict.keys())[1]], smape_dict[list(smape_dict.keys())[3]], smape_dict[list(smape_dict.keys())[4]], smape_dict[list(smape_dict.keys())[5]], smape_dict[list(smape_dict.keys())[6]]]

    with open(file_path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        line = '\t'.join(f'{indi}' for indi in indicator_list) + '\t' + '\t'.join(f'{num:.5f}'for num in metrics_list) + '\n'
        f.writelines(line)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def new_test_metrics(true, pred):

    vps = VPS(true, pred)
    
    _, T, __ = true.shape
    eval_steps = [int(T*0.1), int(T * 0.2), int(T * 0.4), int(T * 0.5), int(T * 0.6), int(T * 0.8), T]
    l2_dict = {}
    for steps in eval_steps:
        l2_dict[steps] = relative_l2_error(true, pred, steps)

    return vps, l2_dict

def pad_and_create_mask(arr, target_T, target_N):
    B, T, N = arr.shape
    padded_arr = np.zeros((B, target_T, target_N), dtype=arr.dtype)
    mask = np.zeros((B, target_T, target_N), dtype=np.float32)
    
    padded_arr[:, :T, :N] = arr
    mask[:, :T, :N] = 1
    
    return padded_arr, mask

def normalize_self(input_time_series, reshape_to_ci=True):

    B, T, N = input_time_series.shape
    means = input_time_series.mean(dim=1, keepdim=True)
    input_time_series = input_time_series - means
    stdev = torch.sqrt(torch.var(input_time_series, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    input_time_series = input_time_series / stdev

    if reshape_to_ci:
        input_time_series = input_time_series.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

    return input_time_series, means, stdev

def denormalize_self(output, means, stdev, from_ci=True):

    if from_ci:
        B, _, N = means.shape
        output = output.reshape(B, N, -1).permute(0, 2, 1)

    output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, output.shape[1], 1))
    output = output + (means[:, 0, :].unsqueeze(1).repeat(1, output.shape[1], 1))

    return output


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"
    
class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5
    

def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)

def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])



def augment(x, y, args):
    import util.augmentation as aug
    augmentation_tags = ""
    if args.jitter:
        x = aug.jitter(x)
        augmentation_tags += "_jitter"
    if args.scaling:
        x = aug.scaling(x)
        augmentation_tags += "_scaling"
    if args.rotation:
        x = aug.rotation(x)
        augmentation_tags += "_rotation"
    if args.permutation:
        x = aug.permutation(x)
        augmentation_tags += "_permutation"
    if args.randompermutation:
        x = aug.permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"
    if args.magwarp:
        x = aug.magnitude_warp(x)
        augmentation_tags += "_magwarp"
    if args.timewarp:
        x = aug.time_warp(x)
        augmentation_tags += "_timewarp"
    if args.windowslice:
        x = aug.window_slice(x)
        augmentation_tags += "_windowslice"
    if args.windowwarp:
        x = aug.window_warp(x)
        augmentation_tags += "_windowwarp"
    if args.spawner:
        x = aug.spawner(x, y)
        augmentation_tags += "_spawner"
    if args.dtwwarp:
        x = aug.random_guided_warp(x, y)
        augmentation_tags += "_rgw"
    if args.shapedtwwarp:
        x = aug.random_guided_warp_shape(x, y)
        augmentation_tags += "_rgws"
    if args.wdba:
        x = aug.wdba(x, y)
        augmentation_tags += "_wdba"
    if args.discdtw:
        x = aug.discriminative_guided_warp(x, y)
        augmentation_tags += "_dgw"
    if args.discsdtw:
        x = aug.discriminative_guided_warp_shape(x, y)
        augmentation_tags += "_dgws"
    return x, augmentation_tags

def run_augmentation_single(x, y, args):
    # print("Augmenting %s"%args.data)
    np.random.seed(args.seed)

    x_aug = x
    y_aug = y


    if len(x.shape)<3:
        # Augmenting on the entire series: using the input data as "One Big Batch"
        #   Before  -   (sequence_length, num_channels)
        #   After   -   (1, sequence_length, num_channels)
        # Note: the 'sequence_length' here is actually the length of the entire series
        x_input = x[np.newaxis,:]
    elif len(x.shape)==3:
        # Augmenting on the batch series: keep current dimension (batch_size, sequence_length, num_channels)
        x_input = x
    else:
        raise ValueError("Input must be (batch_size, sequence_length, num_channels) dimensional")

    if args.augmentation_ratio > 0:
        augmentation_tags = "%d"%args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            x_aug, augmentation_tags = augment(x_input, y, args)
            # print("Round %d: %s done"%(n, augmentation_tags))
        if args.extra_tag:
            augmentation_tags += "_"+args.extra_tag
    else:
        augmentation_tags = args.extra_tag

    if(len(x.shape)<3):
        # Reverse to two-dimensional in whole series augmentation scenario
        x_aug = x_aug.squeeze(0)
    return x_aug, y_aug, augmentation_tags
