import os
import urllib
import pandas as pd
import numpy as np
from tqdm import tqdm
from dysts.flows import *
import dysts
# import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from util import *
from sdeint import itoEuler
from scipy.integrate import solve_ivp
import pickle

def generate_original_data(args):

    if os.path.exists(f'{args.data_dir}/origin.npy') and os.path.exists(f'{args.data_dir}/origin_train.npy') and os.path.exists(f'{args.data_dir}/origin_val.npy') and os.path.exists(f'{args.data_dir}/origin_test.npy'): 
        return
    else:
        os.makedirs(args.data_dir, exist_ok=True)

    if args.generalization:
        pass
    
    else:
        eq = getattr(dysts.flows, args.system)()
        integrator_args = {
            "pts_per_period": args.pts_per_period,
            "atol": args.atol,
            "rtol": args.rtol,
        }
        if args.system == 'Lorenz96':
            eq.ic = np.random.randn(args.variable_num)
        ic_traj = eq.make_trajectory(1000, pts_per_period=10, atol=1e-12, rtol=1e-12)
        np.random.seed(0)
        sel_inds = np.random.choice(range(1000), size=args.num_trajectory, replace=False).astype(int)
        ic_context_test = ic_traj[sel_inds, :]
        trajs_all = []
        for ic in tqdm(ic_context_test, total=len(ic_context_test)):
            eq.ic = np.copy(ic)
            traj = eq.make_trajectory(args.data_training_length + args.data_valid_length + args.data_forecast_length, timescale='Lyapunov', method='Radau', **integrator_args)
            trajs_all.append(traj)

        trajs_all = np.array(trajs_all)
        trajs_train = trajs_all[:, :args.data_training_length, :]
        trajs_val = trajs_all[:, args.data_training_length:args.data_training_length + args.data_valid_length, :]

        # trajs_test_context = trajs_all[:, max(0, args.data_training_length + args.data_valid_length - args.data_forecast_context_length):args.data_training_length + args.data_valid_length, :]
        trajs_test = trajs_all[:, args.data_training_length + args.data_valid_length:, :]

        np.save(f'{args.data_dir}/origin.npy', trajs_all)
        np.save(f'{args.data_dir}/origin_train.npy', trajs_train)
        np.save(f'{args.data_dir}/origin_val.npy', trajs_val)
        np.save(f'{args.data_dir}/origin_test.npy', trajs_test)

        if args.parameter_tuning_data:

            sel_tuning_inds = np.random.choice(np.setdiff1d(range(1000), sel_inds), size=args.num_trajectory_tune)
            ic_context_test_tune = ic_traj[sel_tuning_inds, :]
            trajs_tune_all = []
            for ic in tqdm(ic_context_test_tune, total=len(ic_context_test_tune)):
                eq.ic = np.copy(ic)
                traj = eq.make_trajectory(args.data_training_length + args.data_valid_length + args.data_forecast_length, timescale='Lyapunov', method='Radau', **integrator_args)
                trajs_tune_all.append(traj)
            
            trajs_tune_all = np.array(trajs_tune_all)
            trajs_tune_train = trajs_tune_all[:, :args.data_training_length, :]
            trajs_tune_val = trajs_tune_all[:, args.data_training_length:args.data_training_length + args.data_valid_length, :]
            trajs_tune_test = trajs_tune_all[:, args.data_training_length + args.data_valid_length:, :]

            np.save(f'{args.data_dir}/origin_tune.npy', trajs_tune_all)
            np.save(f'{args.data_dir}/origin_tune_train.npy', trajs_tune_train)
            np.save(f'{args.data_dir}/origin_tune_val.npy', trajs_tune_val)
            np.save(f'{args.data_dir}/origin_tune_test.npy', trajs_tune_test)

        # np.save(f'{args.data_dir}/origin_test_context.npy', trajs_test_context)

# def process_data(args, trajs_all=None, trajs_train=None, trajs_val=None, trajs_test=None, trajs_large=None):
    
#     if args.generalization:
#         pass
    
#     else:
        
#         if not args.large_train:
#             if not args.shift_dataset:
#                 if os.path.exists(f'{args.data_dir}/train/lookback{args.lookback}_predlen{args.horizon_tr}.npz') and \
#                 os.path.exists(f'{args.data_dir}/val/lookback{args.lookback}.npz') and \
#                 os.path.exists(f'{args.data_dir}/test/lookback{args.lookback}.npz'):
#                     return
#                 else:
#                     os.makedirs(f'{args.data_dir}/train', exist_ok=True)
#                     os.makedirs(f'{args.data_dir}/val', exist_ok=True)
#                     os.makedirs(f'{args.data_dir}/test', exist_ok=True)

#             else:
#                 if os.path.exists(f'{args.data_dir}/train/lookback{args.lookback}_shift.npy') and \
#                 os.path.exists(f'{args.data_dir}/val/lookback{args.lookback}.npz') and \
#                 os.path.exists(f'{args.data_dir}/test/lookback{args.lookback}.npz'):
#                     return
#                 else:
#                     os.makedirs(f'{args.data_dir}/train', exist_ok=True)
#                     os.makedirs(f'{args.data_dir}/val', exist_ok=True)
#                     os.makedirs(f'{args.data_dir}/test', exist_ok=True)
#         else:

#             # if not args.shift_dataset:
#             #     if os.path.exists(f'{args.data_dir}/train/lookback{args.lookback}_predlen{args.horizon_tr}_large_train.npz') and \
#             #     os.path.exists(f'{args.data_dir}/val/lookback{args.lookback}.npz') and \
#             #     os.path.exists(f'{args.data_dir}/test/lookback{args.lookback}.npz'):
#             #         return
#             #     else:
#             #         os.makedirs(f'{args.data_dir}/train', exist_ok=True)
#             #         os.makedirs(f'{args.data_dir}/val', exist_ok=True)
#             #         os.makedirs(f'{args.data_dir}/test', exist_ok=True)

#             # else:
                
#             #     if os.path.exists(f'{args.data_dir}/train/lookback{args.lookback}_shift_large_train.npy') and \
#             #     os.path.exists(f'{args.data_dir}/val/lookback{args.lookback}.npz') and \
#             #     os.path.exists(f'{args.data_dir}/test/lookback{args.lookback}.npz'):
#             #         return
                
#             #     else:
#             #         os.makedirs(f'{args.data_dir}/train', exist_ok=True)
#             #         os.makedirs(f'{args.data_dir}/val', exist_ok=True)
#             #         os.makedirs(f'{args.data_dir}/test', exist_ok=True)

#             os.makedirs(f'{args.data_dir}/train', exist_ok=True)
#             os.makedirs(f'{args.data_dir}/val', exist_ok=True)
#             os.makedirs(f'{args.data_dir}/test', exist_ok=True)


#         if not args.shift_dataset:
#             train_x, train_y = [], []
#             for idx, traj in tqdm(enumerate(trajs_train), total=len(trajs_train)):
#                 t_stamp = range(traj.shape[0]-args.lookback-args.horizon_tr)
#                 print(traj.shape[0]-args.lookback-args.horizon_tr)
#                 rand_t_stamp = np.random.permutation(t_stamp)
#                 train_x.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
#                 train_y.append(np.array([traj[i+args.lookback:i+args.lookback+args.horizon_tr] for i in rand_t_stamp]))

#             train_x = np.concatenate(train_x, axis=0)
#             train_y = np.concatenate(train_y, axis=0)

#             if args.large_train:
#                 train_x_large, train_y_large = [], []
#                 for idx, traj in tqdm(enumerate(trajs_large), total=len(trajs_large)):
#                     t_stamp = range(traj.shape[0]-args.lookback-args.horizon_tr)
#                     rand_t_stamp = np.random.permutation(t_stamp)
#                     train_x_large.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
#                     train_y_large.append(np.array([traj[i+args.lookback:i+args.lookback+args.horizon_tr] for i in rand_t_stamp]))

#                 train_x_large = np.concatenate(train_x_large, axis=0)
#                 train_y_large = np.concatenate(train_y_large, axis=0)

#                 train_x = np.concatenate([train_x, train_x_large], axis=0)
#                 train_y = np.concatenate([train_y, train_y_large], axis=0)

#             if not args.large_train:
#                 np.savez(f'{args.data_dir}/train/lookback{args.lookback}_predlen{args.horizon_tr}.npz', x=train_x, y=train_y)
#             else:
#                 np.savez(f'{args.data_dir}/train/lookback{args.lookback}_predlen{args.horizon_tr}_large_train.npz', x=train_x, y=train_y)
        
#         else:
#             train_data = []
#             for idx, traj in tqdm(enumerate(trajs_train), total=len(trajs_train)):
#                 t_stamp = range(traj.shape[0]-args.lookback-args.horizon_tr)
#                 rand_t_stamp = np.random.permutation(t_stamp)

#                 train_data.append(np.array([traj[i:i+args.lookback+args.horizon_tr] for i in rand_t_stamp]))

#             train_data = np.concatenate(train_data, axis=0)
            
#             if args.large_train:
#                 train_data_large = []
#                 for idx, traj in tqdm(enumerate(trajs_large), total=len(trajs_large)):
#                     t_stamp = range(traj.shape[0]-args.lookback-args.horizon_tr)
#                     rand_t_stamp = np.random.permutation(t_stamp)
#                     train_data_large.append(np.array([traj[i:i+args.lookback+args.horizon_tr] for i in rand_t_stamp]))
                
#                 train_data_large = np.concatenate(train_data_large, axis=0)
#                 train_data = np.concatenate([train_data, train_data_large], axis=0)
            

#             if not args.large_train:
#                 np.save(f'{args.data_dir}/train/lookback{args.lookback}_shift.npy', train_data)
#             else:
#                 np.save(f'{args.data_dir}/train/lookback{args.lookback}_shift_large_train.npy', train_data)


#         if os.path.exists(f'{args.data_dir}/val/lookback{args.lookback}.npz') and os.path.exists(f'{args.data_dir}/test/lookback{args.lookback}.npz'):
#             return
#         val_x, val_y = [], []
#         test_x, test_y = [], []

#         for idx, traj in tqdm(enumerate(trajs_val), total=len(trajs_val)):

#             val_context = trajs_all[idx][max(0, args.data_training_length - args.lookback):args.data_training_length]
            
#             val_x.append(np.array(val_context)[None, :]) # (1, context_len, 3)
#             val_y.append(np.array(traj)[None, :]) # (val_len, 3)
            
#         for idx, traj in tqdm(enumerate(trajs_test), total=len(trajs_test)):
            
            
#             # test_context = trajs_all[idx][max(0, args.data_training_length + args.data_valid_length - args.lookback):args.data_training_length + args.data_valid_length, :]
#             test_context = trajs_all[idx][max(0, args.data_training_length - args.lookback):args.data_training_length, :]

#             test_x.append(np.array(test_context)[None, :])
#             test_y.append(np.array(traj)[None, :]) # (1, test_len, 3)
        
#         # concate
#         val_x = np.concatenate(val_x, axis=0)
#         val_y = np.concatenate(val_y, axis=0)
#         test_x = np.concatenate(test_x, axis=0)
#         test_y = np.concatenate(test_y, axis=0)

#         # save
#         np.savez(f'{args.data_dir}/val/lookback{args.lookback}.npz', x=val_x, y=val_y)
#         np.savez(f'{args.data_dir}/test/lookback{args.lookback}.npz', x=test_x, y=test_y)


def process_data(args, trajs_all=None, in_ex_split=False, all_train_trajs=None, all_test_trajs=None):

    train_folder_name, val_folder_name, test_folder_name = 'train', 'val', 'test'

    if args.exp_name != 'standard':
        train_folder_name, val_folder_name, test_folder_name = f'{args.exp_name}_' + train_folder_name, f'{args.exp_name}_' + val_folder_name, f'{args.exp_name}_' + test_folder_name

    # if args.exp_name == 'mi':
    #     train_folder_name, val_folder_name, test_folder_name = 'mi_' + train_folder_name, 'mi_' + val_folder_name, 'mi_' + test_folder_name
    
    # elif args.exp_name == 'mp_full':
    #     train_folder_name, val_folder_name, test_folder_name = 'mp_full_' + train_folder_name, 'mp_full_' + val_folder_name, 'mp_full_' + test_folder_name

    # elif args.exp_name == 'mp_in':


    os.makedirs(f'{args.data_dir}/{train_folder_name}', exist_ok=True)
    os.makedirs(f'{args.data_dir}/{val_folder_name}', exist_ok=True)
    os.makedirs(f'{args.data_dir}/{test_folder_name}', exist_ok=True)

    if args.exp_name == 'mi' or args.exp_name == 'mp_full':
        use_traj_num = range(args.multi_num)
        trajs_all = trajs_all[use_traj_num]

    elif args.exp_name == 'mp_in':
        assert in_ex_split
        assert args.multi_num == 6
        
    elif args.exp_name == 'mp_ex':
        assert in_ex_split
        assert args.multi_num == 6

    if not in_ex_split:
        trajs_normal = trajs_all[:, -1000:, :]
        trajs_normal_train = trajs_normal[:, :550, :]
        trajs_normal_val = trajs_normal[:, 550:700, :]
        trajs_normal_test = trajs_normal[:, 700:, :]
        trajs_large_train = trajs_all[:, :-1000, :]
    else:
        trajs_normal_in_ex_split = all_train_trajs[:, -1000:, :]
        trajs_normal_train = trajs_normal_in_ex_split[:, :550, :]
        trajs_normal_val = trajs_normal_in_ex_split[:, 550:700, :]
        trajs_large_train = all_train_trajs[:, :-1000, :]

        trajs_normal_test_in_ex_split = all_test_trajs[:, -1000:, :]
        test_normal_test_context = trajs_normal_test_in_ex_split[:, :700, :]
        trajs_normal_test = trajs_normal_test_in_ex_split[:, 700:, :]


    if not args.shift_dataset:
        if not args.large_train:
            generate_flag = not os.path.exists(f'{args.data_dir}/{train_folder_name}/lookback{args.lookback}_predlen{args.horizon_tr}.npz')
        else:
            generate_flag = not os.path.exists(f'{args.data_dir}/{train_folder_name}/lookback{args.lookback}_predlen{args.horizon_tr}_large_train_{args.large_train_ratio}.npz')
    else:
        if not args.large_train:
            generate_flag = not os.path.exists(f'{args.data_dir}/{train_folder_name}/lookback{args.lookback}_shift.npy')
        else:
            generate_flag = not os.path.exists(f'{args.data_dir}/{train_folder_name}/lookback{args.lookback}_shift_large_train_{args.large_train_ratio}.npy')
    
    if generate_flag:
        if not args.shift_dataset:
            train_x, train_y = [], []
            for idx, traj in tqdm(enumerate(trajs_normal_train), total=len(trajs_normal_train)):
                t_stamp = range(traj.shape[0]-args.lookback-args.horizon_tr)
                # print(traj.shape[0]-args.lookback-args.horizon_tr)
                rand_t_stamp = np.random.permutation(t_stamp)
                train_x.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
                train_y.append(np.array([traj[i+args.lookback:i+args.lookback+args.horizon_tr] for i in rand_t_stamp]))

            train_x = np.concatenate(train_x, axis=0)
            train_y = np.concatenate(train_y, axis=0)

            if args.large_train:
                train_x_large, train_y_large = [], []
                for idx, traj in tqdm(enumerate(trajs_large_train), total=len(trajs_large_train)):
                    t_stamp = range(traj.shape[0]-args.lookback-args.horizon_tr)
                    rand_t_stamp = np.random.permutation(t_stamp)
                    train_x_large.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
                    train_y_large.append(np.array([traj[i+args.lookback:i+args.lookback+args.horizon_tr] for i in rand_t_stamp]))

                train_x_large = np.concatenate(train_x_large, axis=0)
                train_y_large = np.concatenate(train_y_large, axis=0)
                total_num = train_x_large.shape[0]
                use_num = int(total_num * args.large_train_ratio)
                use_idx = np.random.choice(total_num, use_num, replace=False)
                train_x_large = train_x_large[use_idx]
                train_y_large = train_y_large[use_idx]
                train_x = np.concatenate([train_x, train_x_large], axis=0)
                train_y = np.concatenate([train_y, train_y_large], axis=0)

            if args.exp_name == 'mi' or args.exp_name == 'mp_full' or args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
                
                if args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
                    
                    assert args.relative_times_standard == args.multi_num

                total_data_num = train_x.shape[0]
                use_data_num = int(total_data_num * args.relative_times_standard / args.multi_num)
                use_idxs = np.random.choice(total_data_num, use_data_num, replace=False)
                train_x = train_x[use_idxs]
                train_y = train_y[use_idxs]


            if not args.large_train:
                np.savez(f'{args.data_dir}/{train_folder_name}/lookback{args.lookback}_predlen{args.horizon_tr}.npz', x=train_x, y=train_y)
            else:
                np.savez(f'{args.data_dir}/{train_folder_name}/lookback{args.lookback}_predlen{args.horizon_tr}_large_train_{args.large_train_ratio}.npz', x=train_x, y=train_y)

        else:
            train_data = []
            for idx, traj in tqdm(enumerate(trajs_normal_train), total=len(trajs_normal_train)):
                t_stamp = range(traj.shape[0]-args.lookback)
                rand_t_stamp = np.random.permutation(t_stamp)
                train_data.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))

            train_data = np.concatenate(train_data, axis=0)
            
            if args.large_train:
                train_data_large = []
                for idx, traj in tqdm(enumerate(trajs_large_train), total=len(trajs_large_train)):
                    t_stamp = range(traj.shape[0]-args.lookback)
                    rand_t_stamp = np.random.permutation(t_stamp)
                    train_data_large.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
                
                train_data_large = np.concatenate(train_data_large, axis=0)
                total_num = train_data_large.shape[0]
                use_num = int(total_num * args.large_train_ratio)
                use_idx = np.random.choice(total_num, use_num, replace=False)
                train_data_large = train_data_large[use_idx]
                train_data = np.concatenate([train_data, train_data_large], axis=0)


            if args.exp_name == 'mi' or args.exp_name == 'mp_full' or args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
                
                if args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
                    
                    assert args.relative_times_standard == args.multi_num
                
                total_data_num = train_data.shape[0]
                use_data_num = int(total_data_num * args.relative_times_standard / args.multi_num)
                use_idxs = np.random.choice(total_data_num, use_data_num, replace=False)
                train_data = train_data[use_idxs]
            
            
            if not args.large_train:
                np.save(f'{args.data_dir}/{train_folder_name}/lookback{args.lookback}_shift.npy', train_data)
            else:
                np.save(f'{args.data_dir}/{train_folder_name}/lookback{args.lookback}_shift_large_train_{args.large_train_ratio}.npy', train_data)

    if os.path.exists(f'{args.data_dir}/{val_folder_name}/lookback{args.lookback}.npz') and os.path.exists(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}.npz') and os.path.exists(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}_test_bench.npz'):
        if not args.shift_dataset:
            if os.path.exists(f'{args.data_dir}/{val_folder_name}/lookback{args.lookback}_predlen{args.horizon_tr}.npz'):
                return
        else:
            if os.path.exists(f'{args.data_dir}/{val_folder_name}/lookback{args.lookback}.npy'):
                return
    
    val_x, val_y = [], []
    
    test_x, test_y = [], []
    
    test_bench_x, test_bench_y = [], []
    
    if not args.shift_dataset:
        val_x_short, val_y_short = [], []
    else:
        val_short_data = []

    for idx, traj in tqdm(enumerate(trajs_normal_val), total=len(trajs_normal_val)):

        val_context = trajs_normal_train[idx]
        val_x.append(np.array(val_context)[None, :]) # (1, context_len, 3)
        val_y.append(np.array(traj)[None, :]) # (val_len, 3)

        if not args.shift_dataset:
            t_stamp = range(traj.shape[0]-args.lookback-args.horizon_tr)
            rand_t_stamp = np.random.permutation(t_stamp)
            val_x_short.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
            val_y_short.append(np.array([traj[i+args.lookback:i+args.lookback+args.horizon_tr] for i in rand_t_stamp]))
        else:
            t_stamp = range(traj.shape[0]-args.lookback)
            rand_t_stamp = np.random.permutation(t_stamp)
            val_short_data.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
        
    for idx, traj in tqdm(enumerate(trajs_normal_test), total=len(trajs_normal_test)):
        
        # test_context = trajs_all[idx][max(0, args.data_training_length + args.data_valid_length - args.lookback):args.data_training_length + args.data_valid_length, :]
        # test_context = trajs_all[idx][max(0, args.data_training_length - args.lookback):args.data_training_length, :]
        if not in_ex_split:
            test_context = np.concatenate([trajs_normal_train[idx], trajs_normal_val[idx]], axis=0)
        else:
            test_context = test_normal_test_context[idx]
            
        test_x.append(np.array(test_context)[None, :])
        test_y.append(np.array(traj)[None, :]) # (1, test_len, 3)

    for idx, traj in tqdm(enumerate(trajs_normal_test), total=len(trajs_normal_test)):

        t_stamp = range(traj.shape[0]-args.lookback-args.horizon_tr)
        rand_t_stamp = np.random.permutation(t_stamp)
        test_bench_x.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
        test_bench_y.append(np.array([traj[i+args.lookback:i+args.lookback+args.horizon_tr] for i in rand_t_stamp]))

    
    

    # concate
    val_x = np.concatenate(val_x, axis=0)
    val_y = np.concatenate(val_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    test_bench_x = np.concatenate(test_bench_x, axis=0)
    test_bench_y = np.concatenate(test_bench_y, axis=0)

    if not args.shift_dataset:

        val_x_short = np.concatenate(val_x_short, axis=0)
        val_y_short = np.concatenate(val_y_short, axis=0)

        if args.exp_name == 'mi' or args.exp_name == 'mp_full' or args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
                
            if args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
                
                assert args.relative_times_standard == args.multi_num
            total_data_num = val_x_short.shape[0]
            use_data_num = int(total_data_num * args.relative_times_standard / args.multi_num)
            use_idxs = np.random.choice(total_data_num, use_data_num, replace=False)
            val_x_short = val_x_short[use_idxs]
            val_y_short = val_y_short[use_idxs]

        np.savez(f'{args.data_dir}/{val_folder_name}/short_lookback{args.lookback}_predlen{args.horizon_tr}.npz', x=val_x_short, y=val_y_short)
    
    else:

        val_short_data = np.concatenate(val_short_data, axis=0)
        if args.exp_name == 'mi' or args.exp_name == 'mp_full' or args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
                
            if args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
                
                assert args.relative_times_standard == args.multi_num
            total_data_num = val_short_data.shape[0]
            use_data_num = int(total_data_num * args.relative_times_standard / args.multi_num)
            use_idxs = np.random.choice(total_data_num, use_data_num, replace=False)
            val_short_data = val_short_data[use_idxs]
        np.save(f'{args.data_dir}/{val_folder_name}/short_lookback{args.lookback}_shift.npy', val_short_data)

    # save
    np.savez(f'{args.data_dir}/{val_folder_name}/lookback{args.lookback}.npz', x=val_x, y=val_y)
    np.savez(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}.npz', x=test_x, y=test_y)
    np.savez(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}_test_bench.npz', x=test_bench_x, y=test_bench_y)



def process_multistep_data(args, trajs_all):

    train_folder_name, val_folder_name, test_folder_name = 'train', 'val', 'test'
    
    # if args.exp_name == 'mi':
    #     train_folder_name, val_folder_name, test_folder_name = 'mi_' + train_folder_name, 'mi_' + val_folder_name, 'mi_' + test_folder_name

    if args.exp_name != 'standard':
        train_folder_name, val_folder_name, test_folder_name = f'{args.exp_name}_' + train_folder_name, f'{args.exp_name}_' + val_folder_name, f'{args.exp_name}_' + test_folder_name

    os.makedirs(f'{args.data_dir}/{train_folder_name}', exist_ok=True)
    os.makedirs(f'{args.data_dir}/{val_folder_name}', exist_ok=True)
    os.makedirs(f'{args.data_dir}/{test_folder_name}', exist_ok=True)

    if args.exp_name == 'mi' or args.exp_name == 'mp_full':
        use_traj_num = range(args.multi_num)
        trajs_all = trajs_all[use_traj_num]

    elif args.exp_name == 'mp_in':
        assert args.multi_num == 6
        
    elif args.exp_name == 'mp_ex':
        assert args.multi_num == 6
    

    trajs_normal = trajs_all[:, -1000:, :]
    trajs_normal_train = trajs_normal[:, :550, :]
    trajs_normal_val = trajs_normal[:, 550:700, :]

    trajs_large_train = trajs_all[:, :-1000, :]



    if not args.large_train:
        generate_flag = not os.path.exists(f'{args.data_dir}/{train_folder_name}/sf_lookback{args.lookback}_predlen{args.horizon_multistep}.npz')
    else:
        generate_flag = not os.path.exists(f'{args.data_dir}/{train_folder_name}/sf_lookback{args.lookback}_predlen{args.horizon_multistep}_large_train_{args.large_train_ratio}.npz')

    if generate_flag:
        train_x, train_y = [], []
        for idx, traj in tqdm(enumerate(trajs_normal_train), total=len(trajs_normal_train)):
            t_stamp = range(traj.shape[0]-args.lookback-args.horizon_multistep)
            # print(traj.shape[0]-args.lookback-args.horizon_multistep)
            rand_t_stamp = np.random.permutation(t_stamp)
            train_x.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
            train_y.append(np.array([traj[i+args.lookback:i+args.lookback+args.horizon_multistep] for i in rand_t_stamp]))

        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)

        if args.large_train:
            train_x_large, train_y_large = [], []
            for idx, traj in tqdm(enumerate(trajs_large_train), total=len(trajs_large_train)):
                t_stamp = range(traj.shape[0]-args.lookback-args.horizon_multistep)
                rand_t_stamp = np.random.permutation(t_stamp)
                train_x_large.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
                train_y_large.append(np.array([traj[i+args.lookback:i+args.lookback+args.horizon_multistep] for i in rand_t_stamp]))

            train_x_large = np.concatenate(train_x_large, axis=0)
            train_y_large = np.concatenate(train_y_large, axis=0)
            total_num = train_x_large.shape[0]
            use_num = int(total_num * args.large_train_ratio)
            use_idx = np.random.choice(total_num, use_num, replace=False)
            train_x_large = train_x_large[use_idx]
            train_y_large = train_y_large[use_idx]
            train_x = np.concatenate([train_x, train_x_large], axis=0)
            train_y = np.concatenate([train_y, train_y_large], axis=0)

        if args.exp_name == 'mi' or args.exp_name == 'mp_full' or args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
                
            if args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
                
                assert args.relative_times_standard == args.multi_num
            
            total_data_num = train_x.shape[0]
            use_data_num = int(total_data_num * args.relative_times_standard / args.multi_num)
            use_idxs = np.random.choice(total_data_num, use_data_num, replace=False)
            train_x = train_x[use_idxs]
            train_y = train_y[use_idxs]

        if not args.large_train:
            np.savez(f'{args.data_dir}/{train_folder_name}/sf_lookback{args.lookback}_predlen{args.horizon_multistep}.npz', x=train_x, y=train_y)
        else:
            np.savez(f'{args.data_dir}/{train_folder_name}/sf_lookback{args.lookback}_predlen{args.horizon_multistep}_large_train_{args.large_train_ratio}.npz', x=train_x, y=train_y)

    if os.path.exists(f'{args.data_dir}/{val_folder_name}/sfval_short_lookback{args.lookback}_predlen{args.horizon_multistep}.npz'):
        return

    val_x_short, val_y_short = [], []

    for idx, traj in tqdm(enumerate(trajs_normal_val), total=len(trajs_normal_val)):

        t_stamp = range(traj.shape[0]-args.lookback-args.horizon_multistep)
        rand_t_stamp = np.random.permutation(t_stamp)
        val_x_short.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
        val_y_short.append(np.array([traj[i+args.lookback:i+args.lookback+args.horizon_multistep] for i in rand_t_stamp]))

    val_x_short = np.concatenate(val_x_short, axis=0)
    val_y_short = np.concatenate(val_y_short, axis=0)

    if args.exp_name == 'mi' or args.exp_name == 'mp_full' or args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
                
        if args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':
            
            assert args.relative_times_standard == args.multi_num
        
        total_data_num = val_x_short.shape[0]
        use_data_num = int(total_data_num * args.relative_times_standard / args.multi_num)
        use_idxs = np.random.choice(total_data_num, use_data_num, replace=False)
        val_x_short = val_x_short[use_idxs]
        val_y_short = val_y_short[use_idxs]

    np.savez(f'{args.data_dir}/{val_folder_name}/sfval_short_lookback{args.lookback}_predlen{args.horizon_multistep}.npz', x=val_x_short, y=val_y_short)


def process_mulinit_test_data(args, multi_trajs_all, ratio_mode=False):

    if args.system != 'EEG':
        assert args.long_test_length == 300
    
    train_folder_name, val_folder_name, test_folder_name = 'train', 'val', 'test'
    if args.exp_name != 'standard':
        train_folder_name, val_folder_name, test_folder_name = f'{args.exp_name}_' + train_folder_name, f'{args.exp_name}_' + val_folder_name, f'{args.exp_name}_' + test_folder_name

    os.makedirs(f'{args.data_dir}/{train_folder_name}', exist_ok=True)
    os.makedirs(f'{args.data_dir}/{val_folder_name}', exist_ok=True)
    os.makedirs(f'{args.data_dir}/{test_folder_name}', exist_ok=True)

    total_trajs = multi_trajs_all.shape[0]

    if args.system != 'ECG' and args.system != 'EEG':
        if not ratio_mode:
            if os.path.exists(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}_trajs_num10.npz'):
                return
        else:
            if os.path.exists(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}_trajs_num{int(total_trajs * 0.1)}.npz'):
                return
            else:
                print('Warning!!!!!!!!!')
            
    elif args.system == 'ECG':
        if os.path.exists(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}_trajs_num20.npz'):
            return
    
    elif args.system == 'EEG':

        if args.long_test_length == 300:

            if os.path.exists(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}_trajs_num24.npz'):
                return
            
        elif args.long_test_length == 600:
            
            if os.path.exists(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}_trajs_num13_600.npz'):
                return



    else:
        raise NotImplementedError

    if not ratio_mode:
        if args.system == 'ECG':
            test_traj_nums = np.array([20])
        elif args.system == 'EEG':
            if args.long_test_length == 300:
                test_traj_nums = np.array([24])
            elif args.long_test_length == 600:
                test_traj_nums = np.array([13])
        else:
            test_traj_nums = np.array([10, 30, 50, 70, 100])
    
    else:
        ratio = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
        test_traj_nums = np.array([int(total_trajs * r) for r in ratio])

    for test_traj_num in test_traj_nums:

        use_traj_idxs = np.random.choice(total_trajs, test_traj_num, replace=False)

        trajs_normal = multi_trajs_all[use_traj_idxs, -1000:, :]
        
        if args.system == 'EEG':
            trajs_normal_test = trajs_normal[:, -args.long_test_length:, :]
        else:
            trajs_normal_test = trajs_normal[:, -300:, :]

        test_x, test_y = [], []


        for idx, traj in tqdm(enumerate(trajs_normal_test), total=len(trajs_normal_test)):
            
            # test_context = trajs_all[idx][max(0, args.data_training_length + args.data_valid_length - args.lookback):args.data_training_length + args.data_valid_length, :]
            # test_context = trajs_all[idx][max(0, args.data_training_length - args.lookback):args.data_training_length, :]

            if args.system == 'EEG':
                test_context = trajs_normal[idx, :-args.long_test_length, :]
            else:
                test_context = trajs_normal[idx, :-300, :]

            test_x.append(np.array(test_context)[None, :])
            test_y.append(np.array(traj)[None, :]) # (1, test_len, 3)

        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)
    
        if args.system == 'EEG' and args.long_test_length == 600:
            np.savez(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}_trajs_num{test_traj_num}_600.npz', x=test_x, y=test_y)
        else:
            np.savez(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}_trajs_num{test_traj_num}.npz', x=test_x, y=test_y)

def process_ood_test_data(args, multi_trajs_all):

    train_folder_name, val_folder_name, test_folder_name = 'train', 'val', 'test'

    if args.exp_name == 'mi':
        train_folder_name, val_folder_name, test_folder_name = 'mi_' + train_folder_name, 'mi_' + val_folder_name, 'mi_' + test_folder_name
    else:
        raise NotImplementedError
    
    os.makedirs(f'{args.data_dir}/{train_folder_name}', exist_ok=True)
    os.makedirs(f'{args.data_dir}/{val_folder_name}', exist_ok=True)
    os.makedirs(f'{args.data_dir}/{test_folder_name}', exist_ok=True)

    if os.path.exists(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}_ood_test.npz'):
        return
    
    test_traj_num = args.ood_test_traj_num
    
    trajs_normal = multi_trajs_all[-test_traj_num:, -1000:, :]
    trajs_normal_test = trajs_normal[:, 700:, :]

    test_x, test_y = [], []
    for idx, traj in tqdm(enumerate(trajs_normal_test), total=len(trajs_normal_test)):
        
        # test_context = trajs_all[idx][max(0, args.data_training_length + args.data_valid_length - args.lookback):args.data_training_length + args.data_valid_length, :]
        # test_context = trajs_all[idx][max(0, args.data_training_length - args.lookback):args.data_training_length, :]
        test_context = trajs_normal[idx, :700, :]
        test_x.append(np.array(test_context)[None, :])
        test_y.append(np.array(traj)[None, :]) # (1, test_len, 3)

    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    np.savez(f'{args.data_dir}/{test_folder_name}/lookback{args.lookback}_ood_test.npz', x=test_x, y=test_y)


def process_data_real(args, trajs_train, trajs_test):

    os.makedirs(f'{args.data_dir}/train', exist_ok=True)
    os.makedirs(f'{args.data_dir}/val', exist_ok=True)
    os.makedirs(f'{args.data_dir}/test', exist_ok=True)

    total_length = trajs_train.shape[0]
    train_length = int(total_length * 0.8)
    
    
    trajs_normal_train = trajs_train[None, :train_length, :]
    trajs_normal_val = trajs_train[None, train_length:, :]
    trajs_normal_test = trajs_test[None, :, :]


    generate_flag = not os.path.exists(f'{args.data_dir}/train/lookback{args.lookback}_shift.npy')
    
    if generate_flag:
        train_data = []
        for idx, traj in tqdm(enumerate(trajs_normal_train), total=len(trajs_normal_train)):
            t_stamp = range(traj.shape[0]-args.lookback)
            rand_t_stamp = np.random.permutation(t_stamp)
            train_data.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))

        train_data = np.concatenate(train_data, axis=0)

        np.save(f'{args.data_dir}/train/lookback{args.lookback}_shift.npy', train_data)


    if os.path.exists(f'{args.data_dir}/val/lookback{args.lookback}.npz') and os.path.exists(f'{args.data_dir}/test/lookback{args.lookback}.npz'):

        if os.path.exists(f'{args.data_dir}/val/lookback{args.lookback}_shift.npy'):
            return
    
    val_x, val_y = [], []
    
    test_x, test_y = [], []
    
    val_short_data = []

    for idx, traj in tqdm(enumerate(trajs_normal_val), total=len(trajs_normal_val)):

        val_context = trajs_normal_train[idx]
        val_x.append(np.array(val_context)[None, :]) # (1, context_len, 3)
        val_y.append(np.array(traj)[None, :]) # (val_len, 3)

        t_stamp = range(traj.shape[0]-args.lookback)
        rand_t_stamp = np.random.permutation(t_stamp)
        val_short_data.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
        
    for idx, traj in tqdm(enumerate(trajs_normal_test), total=len(trajs_normal_test)):
        
        # test_context = trajs_all[idx][max(0, args.data_training_length + args.data_valid_length - args.lookback):args.data_training_length + args.data_valid_length, :]
        # test_context = trajs_all[idx][max(0, args.data_training_length - args.lookback):args.data_training_length, :]
        # test_context = np.concatenate([trajs_normal_train[idx], trajs_normal_val[idx]], axis=0)
        total_traj_len = traj.shape[0]
        context_len = int(total_traj_len * 0.5)
        test_context = traj[:context_len]
        test_x.append(np.array(test_context)[None, :])
        test_y.append(np.array(traj)[None, context_len:, ]) # (1, test_len, 3)
    
    # concate
    val_x = np.concatenate(val_x, axis=0)
    val_y = np.concatenate(val_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)


    val_short_data = np.concatenate(val_short_data, axis=0)
    np.save(f'{args.data_dir}/val/short_lookback{args.lookback}_shift.npy', val_short_data)

    # save
    np.savez(f'{args.data_dir}/val/lookback{args.lookback}.npz', x=val_x, y=val_y)
    np.savez(f'{args.data_dir}/test/lookback{args.lookback}.npz', x=test_x, y=test_y)

def process_multistep_real_data(args, trajs_train, trajs_test):

    os.makedirs(f'{args.data_dir}/train', exist_ok=True)
    os.makedirs(f'{args.data_dir}/val', exist_ok=True)
    os.makedirs(f'{args.data_dir}/test', exist_ok=True)

    total_length = trajs_train.shape[0]
    train_length = int(total_length * 0.8)
    
    trajs_normal_train = trajs_train[None, :train_length, :]
    trajs_normal_val = trajs_train[None, train_length:, :]
    trajs_normal_test = trajs_test[None, :, :]

    generate_flag = not os.path.exists(f'{args.data_dir}/train/sf_lookback{args.lookback}_predlen{args.horizon_multistep}.npz')

    if generate_flag:
        train_x, train_y = [], []
        for idx, traj in tqdm(enumerate(trajs_normal_train), total=len(trajs_normal_train)):
            t_stamp = range(traj.shape[0]-args.lookback-args.horizon_multistep)
            # print(traj.shape[0]-args.lookback-args.horizon_multistep)
            rand_t_stamp = np.random.permutation(t_stamp)
            train_x.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
            train_y.append(np.array([traj[i+args.lookback:i+args.lookback+args.horizon_multistep] for i in rand_t_stamp]))

        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)

        np.savez(f'{args.data_dir}/train/sf_lookback{args.lookback}_predlen{args.horizon_multistep}.npz', x=train_x, y=train_y)
        
    if os.path.exists(f'{args.data_dir}/val/sfval_short_lookback{args.lookback}_predlen{args.horizon_multistep}.npz'):
        return

    val_x_short, val_y_short = [], []

    for idx, traj in tqdm(enumerate(trajs_normal_val), total=len(trajs_normal_val)):

        t_stamp = range(traj.shape[0]-args.lookback-args.horizon_multistep)
        rand_t_stamp = np.random.permutation(t_stamp)
        val_x_short.append(np.array([traj[i:i+args.lookback] for i in rand_t_stamp]))
        val_y_short.append(np.array([traj[i+args.lookback:i+args.lookback+args.horizon_multistep] for i in rand_t_stamp]))

    val_x_short = np.concatenate(val_x_short, axis=0)
    val_y_short = np.concatenate(val_y_short, axis=0)

    np.savez(f'{args.data_dir}/val/sfval_short_lookback{args.lookback}_predlen{args.horizon_multistep}.npz', x=val_x_short, y=val_y_short)
    