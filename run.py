import os
import time
import signal
import argparse
from multiprocessing import Process
import multiprocessing
from data.generator import *
from util import *
import setproctitle
from exp.exp_forecasting import Exp_Long_Term_Forecast
import dysts

from args.args import get_args
from args.suffix import get_suffix
from data_prepare import Data_Generate
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['TRITON_DEBUG'] = '1'

setproctitle.setproctitle('@liuchang')

if __name__ == '__main__':

    args = get_args()

    assert args.exp_name in ['standard']

    if  (args.system == 'Weather') or (args.system == 'traffic') or (args.system == 'electricity') or (args.system == 'ETTh1') or (args.system == 'ETTh2') or (args.system == 'ETTm1') or (args.system == 'ETTm2') or (args.system == 'exchange_rate'):
        args.data_type = 'real'
    
    elif (args.system == 'Lorenz') or (args.system == 'Rossler') or (args.system == 'Lorenz96') or (args.system == 'ECG') or (args.system == 'EEG'):
        args.data_type = 'sim'
    
    else:
        raise ValueError('Unknown system!')
    
    # if args.model == 'Mamba_MTP':
    #     assert args.mtp_steps > 0
    if not args.mix_dataset:
        args.data_dir = os.path.join(args.data_dir_base, args.system)
    
    # if args.adjust_resolution:
    #     args.data_dir = args.data_dir + f'_{args.dt}'
    # if (args.system != 'ECG') and (args.system != 'EEG'):
    #     args.data_dir = args.data_dir + f'_{args.pts_per_period}'

    if (args.data_type == 'sim') and (args.system != 'ECG') and (args.system != 'EEG'):
        args.data_dir = args.data_dir + f'_{args.pts_per_period}'

    if args.generalization:
        args.data_dir = os.path.join(args.data_dir, 'generalization')
    args.allow_gpu_list = eval(args.allow_gpu_list)

    # if 'Lorenz' in args.system:
    #     args.variable_num = Lorenz().ic.shape[-1]
    #     args.grids_dim = 1

    if args.data_type == 'sim':

        if not args.partial_observe:

            try:
                args.variable_num = getattr(dysts.flows, args.system)().ic.shape[-1]
            except:
                if args.system == 'Lorenz_RC':
                    args.variable_num = 3
                    print('Manually setting dim for Lorenz_RC')
                elif args.system == 'EEG':
                    args.variable_num = 64
                elif args.system == 'ECG':
                    args.variable_num = 5
                    
            if args.system == 'Lorenz96':
                args.variable_num = 5
        else:
            assert args.system == 'Lorenz'
            args.variable_num = args.partial_observe_dim

    else:
        data_path = os.path.join(args.data_file_path, f'{args.system}', f'{args.system}.csv')
        df_raw = pd.read_csv(data_path)
        cols = list(df_raw.columns)
        cols.remove('OT')
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + ['OT']]
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        df_data = df_data.values
        args.variable_num = df_data.shape[1]
    
    # elif 'HindmarshRose' in args.system:
    #     args.variable_num = HindmarshRose().ic.shape[-1]
    #     args.grids_dim = 1
    # elif 'Lorenz96' in args.system:
    #     args.variable_num = Lorenz96().ic.shape[-1]
    #     args.grids_dim = 1
    # elif 'Rossler' in args.system:
    #     args.variable_num = Rossler().ic.shape[-1]
    #     args.grids_dim = 1
    # elif 'L96' in args.system:
    #     args.variable_num = args.system_dim
    #     args.grids_dim = 1

    if args.model == 'ours':
        print('Select our model MambaLM.')
        if args.model_size == 'tiny': # 2.41 MB
            args.d_model = 256
            args.n_layers = 2

        elif args.model_size == 'mini': # 7.56 MB
            args.d_model = 256
            args.n_layers = 8
        
        elif args.model_size == 'small': # 11.67 MB
            args.d_model = 512
            args.n_layers = 4
        
        elif args.model_size == 'base': # 32.54 MB
            args.d_model = 768
            args.n_layers = 6

        elif args.model_size == 'medium': # 69.75 MB
            args.d_model = 1024
            args.n_layers = 8

        elif args.model_size == 'large': # 246.19 MB
            args.d_model = 1536
            args.n_layers = 14

        elif args.model_size == 'use_parameter':
            pass
        else:
            raise NotImplementedError
    # else:
    #     print('Do not have pre-defined model size.')

    suffix = get_suffix(args)
    
    # if args.adjust_resolution:
    if args.data_type == 'sim':
        args.log_dir = f'./logs/{args.system}_{args.pts_per_period}/{args.model}/' + suffix
    elif args.data_type == 'real':
        args.log_dir = f'./logs/{args.system}/{args.model}/' + suffix
    else:
        raise ValueError('Unknown type!')
    # args.log_dir = f'./logs/{args.system}_{args.dt}/{args.model}/w{args.window_size}_t{args.token_size}'

    # else:
    #     args.log_dir = f'./logs/{args.system}/{args.model}/' + suffix
    #     # args.log_dir = f'./logs/{args.system}/{args.model}/w{args.window_size}_t{args.token_size}'
    if (not args.no_check_data) and (args.data_type == 'sim'):
        Data_Generate(args)

    if args.device == 'cuda':
        if not args.external_parallel_gpu:
            gpu_controller = AutoGPU(args.memory_size, args.allow_gpu_list)
        
            gpu_id = gpu_controller.choice_gpu() if args.device == 'cuda' else 0

        else:
            gpu_id = 0
    else:
        gpu_id = -1

    random_seed = args.random_seed
    seed_everything(random_seed)
    set_cpu_num(1)

    if args.device == 'cuda':
        torch.cuda.set_device(gpu_id)

    if gpu_id == -1:
        args.device = 'cpu'
    print('Args in experiment:')
    print(args)

        # 
    assert args.model == 'Mamba_MTP'
    exp = Exp_Long_Term_Forecast(args)

    if not args.test_only:
        print('>>>>>> Start Training! >>>>>>>')
        exp.train(is_print=True, random_seed=random_seed)

    print('>>>>>> Start Testing! >>>>>>>')
  
    exp.test(is_print=True, random_seed=random_seed)

    if args.device == 'cuda':
        torch.cuda.empty_cache()
    

