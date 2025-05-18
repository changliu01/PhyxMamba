import os
import numpy as np
from data.generator import process_data, process_multistep_data, process_mulinit_test_data, process_data_real, process_multistep_real_data, process_ood_test_data
import pickle


def Data_Generate(args):

    # if args.system == 'ECG' or args.system == 'EEG':
        
    #     traj_train_path = os.path.join(args.data_dir, 'train.npy')
    #     traj_test_path = os.path.join(args.data_dir, 'test.npy')
    #     trajs_train = np.load(traj_train_path, allow_pickle=True)
    #     trajs_test = np.load(traj_test_path, allow_pickle=True)
    #     print('Processing real data')
    #     process_data_real(args, trajs_train=trajs_train, trajs_test=trajs_test)
    #     process_multistep_real_data(args, trajs_train=trajs_train, trajs_test=trajs_test)

    # else:

    if args.exp_name == 'standard':
        traj_all_path = os.path.join(args.data_dir, 'long_traj.npy')
        trajs_all = np.load(traj_all_path, allow_pickle=True)
        if args.system == 'ECG':
            multi_trajs_all_path = os.path.join(args.data_dir, 'mult_initial_20.npy')
        elif args.system == 'EEG':
            # if args.long_test_length == 600:
            multi_trajs_all_path = os.path.join(args.data_dir, 'mult_initial_13_600.npy')
            # elif args.long_test_length == 300:
                # raise 
                # multi_trajs_all_path = os.path.join(args.data_dir, 'mult_initial_24.npy')
            # else:
            #     raise NotImplementedError
        else:
            multi_trajs_all_path = os.path.join(args.data_dir, 'mult_initial_100.npy')
        multi_trajs_all = np.load(multi_trajs_all_path, allow_pickle=True)
        print('Processing data')
        process_data(args, trajs_all=trajs_all)
        process_multistep_data(args, trajs_all=trajs_all)
        process_mulinit_test_data(args, multi_trajs_all=multi_trajs_all)

    elif args.exp_name == 'mi':

        traj_all_path = os.path.join(args.data_dir, 'mult_initial_100.npy')
        trajs_all = np.load(traj_all_path, allow_pickle=True)
        print('Processing data')
        process_data(args, trajs_all=trajs_all)
        process_multistep_data(args, trajs_all=trajs_all)
        process_ood_test_data(args, multi_trajs_all=trajs_all)
    
    elif args.exp_name == 'mp_full':

        lyap_info = pickle.load(open(os.path.join(args.data_dir, 'system_lyap.pkl'), 'rb'))
        keys = list(lyap_info.keys())
        trajs_all = []
        for key in keys:
            traj_all_path = os.path.join(args.data_dir, 'transfer_params', str(key), 'long_traj.npy')
            trajs_all.append(np.load(traj_all_path, allow_pickle=True))
        trajs_all = np.concatenate(trajs_all, axis=0)

        multi_trajs_all = []
        for key in keys:
            multi_traj_all_path = os.path.join(args.data_dir, 'transfer_params', str(key), 'mult_initial_100.npy')
            multi_trajs_all.append(np.load(multi_traj_all_path, allow_pickle=True))
        multi_trajs_all = np.concatenate(multi_trajs_all, axis=0)
        print('Processing data')
        process_data(args, trajs_all=trajs_all)
        process_multistep_data(args, trajs_all=trajs_all)
        process_mulinit_test_data(args, multi_trajs_all=multi_trajs_all, ratio_mode=True)

    elif args.exp_name == 'mp_in' or args.exp_name == 'mp_ex':

        lyap_info = pickle.load(open(os.path.join(args.data_dir, 'system_lyap.pkl'), 'rb'))
        keys = list(lyap_info.keys())
        all_train_trajs = []
        all_test_trajs = []
        total_idxs = np.arange(len(keys))
        test_idxs = np.array([2, 4, 6]) if args.exp_name == 'mp_in' else np.array([0, 7, 8])
        train_idxs = np.setdiff1d(total_idxs, test_idxs)

        for idx in train_idxs:
            traj_all_path = os.path.join(args.data_dir, 'transfer_params', str(keys[idx]), 'long_traj.npy')
            all_train_trajs.append(np.load(traj_all_path, allow_pickle=True))
        all_train_trajs = np.concatenate(all_train_trajs, axis=0)
        
        for idx in test_idxs:
            traj_all_path = os.path.join(args.data_dir, 'transfer_params', str(keys[idx]), 'long_traj.npy')
            all_test_trajs.append(np.load(traj_all_path, allow_pickle=True))
        all_test_trajs = np.concatenate(all_test_trajs, axis=0)
        
        multi_trajs_all = []
        for idx in test_idxs:
            multi_traj_all_path = os.path.join(args.data_dir, 'transfer_params', str(keys[idx]), 'mult_initial_100.npy')
            multi_trajs_all.append(np.load(multi_traj_all_path, allow_pickle=True))
        multi_trajs_all = np.concatenate(multi_trajs_all, axis=0)
        
        print('Processing data')
        process_data(args, in_ex_split=True, all_train_trajs=all_train_trajs, all_test_trajs=all_test_trajs)
        process_multistep_data(args, all_train_trajs)
        process_mulinit_test_data(args, multi_trajs_all=multi_trajs_all, ratio_mode=True)

    elif args.exp_name == 'contrastive_context_full':

        # if os.path.exists(os.path.join(args.data_dir, args.exp_name, 'train.npy')) and \
        #     os.path.exists(os.path.join(args.data_dir, args.exp_name, 'val.npy')) and \
        #     os.path.exists(os.path.join(args.data_dir, args.exp_name, 'test.npy')):
        #     return
            
        lyap_info = pickle.load(open(os.path.join(args.data_dir, 'system_lyap.pkl'), 'rb'))
        keys = list(lyap_info.keys())
        total_idxs = np.arange(len(keys))

        all_trajs = []
        for idx in total_idxs:
            traj_all_path = os.path.join(args.data_dir, 'transfer_params', str(keys[idx]), 'long_traj.npy')
            trajs_all = np.load(traj_all_path, allow_pickle=True)
            all_trajs.append(trajs_all)
        
        all_trajs = np.concatenate(all_trajs, axis=0) # (n_env 9, 30000, 3)

        # train_len = int(all_trajs.shape[1] * 0.8)
        # val_len = int(all_trajs.shape[1] * 0.1)
        # test_len = all_trajs.shape[1] - train_len - val_len

        # trajs_train = all_trajs[:, :train_len]
        # trajs_val = all_trajs[:, train_len:train_len+val_len]
        # trajs_test = all_trajs[:, train_len+val_len:]

        all_trajs = all_trajs[:, -args.lookback:, :]
        trajs_train = all_trajs[:, :, :]
        trajs_val = all_trajs[:, :, :]
        trajs_test = all_trajs[:, :, :]


        os.makedirs(os.path.join(args.data_dir, args.exp_name), exist_ok=True)

        np.save(os.path.join(args.data_dir, args.exp_name, 'train.npy'), trajs_train)
        np.save(os.path.join(args.data_dir, args.exp_name, 'val.npy'), trajs_val)
        np.save(os.path.join(args.data_dir, args.exp_name, 'test.npy'), trajs_test)

    elif args.exp_name == 'contrastive_context_in' or args.exp_name == 'contrastive_context_ex':

        # if os.path.exists(os.path.join(args.data_dir, args.exp_name, 'train.npy')) and \
        #     os.path.exists(os.path.join(args.data_dir, args.exp_name, 'val.npy')) and \
        #     os.path.exists(os.path.join(args.data_dir, args.exp_name, 'test.npy')):
        #     return
        
        lyap_info = pickle.load(open(os.path.join(args.data_dir, 'system_lyap.pkl'), 'rb'))
        keys = list(lyap_info.keys())
        total_idxs = np.arange(len(keys))

        test_idxs = np.array([2, 4, 6]) if args.exp_name == 'contrastive_context_in' else np.array([0, 7, 8])
        train_idxs = np.setdiff1d(total_idxs, test_idxs)

        train_trajs = []
        test_trajs = []
        
        for idx in train_idxs:
            traj_all_path = os.path.join(args.data_dir, 'transfer_params', str(keys[idx]), 'long_traj.npy')
            trajs_all = np.load(traj_all_path, allow_pickle=True)
            train_trajs.append(trajs_all)
        train_trajs = np.concatenate(train_trajs, axis=0)

        for idx in test_idxs:
            traj_all_path = os.path.join(args.data_dir, 'transfer_params', str(keys[idx]), 'long_traj.npy')
            trajs_all = np.load(traj_all_path, allow_pickle=True)
            test_trajs.append(trajs_all)
        test_trajs = np.concatenate(test_trajs, axis=0)

        train_len = int(train_trajs.shape[1] * 0.8)
        val_len = int(train_trajs.shape[1] * 0.1)
        test_len = train_trajs.shape[1] - train_len - val_len

        trajs_train = train_trajs[:, :train_len]
        trajs_val = train_trajs[:, train_len:train_len+val_len]
        trajs_test = train_trajs[:, -test_len:]

        os.makedirs(os.path.join(args.data_dir, args.exp_name), exist_ok=True)

        np.save(os.path.join(args.data_dir, args.exp_name, 'train.npy'), trajs_train)
        np.save(os.path.join(args.data_dir, args.exp_name, 'val.npy'), trajs_val)
        np.save(os.path.join(args.data_dir, args.exp_name, 'test.npy'), trajs_test)

    else:
        raise NotImplementedError


# def Data_Generate(args):

#     if args.mix_dataset:
#         return
    
#     if args.generalization:
#         return
    
#     else:
#         # try:
#         #     print('Loading original simulation data')
            
#         #     trajs_all_path = os.path.join(args.data_dir, 'origin.npy')
#         #     traj_train_path, traj_val_path, traj_test_path = os.path.join(args.data_dir, 'origin_train.npy'), os.path.join(args.data_dir, 'origin_val.npy'), os.path.join(args.data_dir, 'origin_test.npy')

#         #     trajs_all = np.load(trajs_all_path, allow_pickle=True)
#         #     trajs_train, trajs_val, trajs_test = np.load(traj_train_path, allow_pickle=True), np.load(traj_val_path, allow_pickle=True), np.load(traj_test_path, allow_pickle=True)

#         # except:
#         #     print('Generating original simulation data')
#         #     generate_original_data(args)

#         #     trajs_all = np.load(trajs_all_path, allow_pickle=True)
#         #     trajs_train, trajs_val, trajs_test = np.load(traj_train_path, allow_pickle=True), np.load(traj_val_path, allow_pickle=True), np.load(traj_test_path, allow_pickle=True)

#         # print('Processing data')
#         # process_data(args, trajs_all=trajs_all, trajs_train=trajs_train, trajs_val=trajs_val, trajs_test=trajs_test)
        
#         if not args.large_train:
#             if not args.shift_dataset:
#                 if os.path.exists(f'{args.data_dir}/train/lookback{args.lookback}_predlen{args.horizon_tr}.npz') and \
#                     os.path.exists(f'{args.data_dir}/val/lookback{args.lookback}.npz') and \
#                     os.path.exists(f'{args.data_dir}/test/lookback{args.lookback}.npz'):
#                         return
                
#             else:
#                 if os.path.exists(f'{args.data_dir}/train/lookback{args.lookback}_shift.npz') and \
#                     os.path.exists(f'{args.data_dir}/val/lookback{args.lookback}.npz') and \
#                     os.path.exists(f'{args.data_dir}/test/lookback{args.lookback}.npz'):
#                         return
#         else:
#             if not args.shift_dataset:
#                 if os.path.exists(f'{args.data_dir}/train/lookback{args.lookback}_predlen{args.horizon_tr}_large_train.npz') and \
#                     os.path.exists(f'{args.data_dir}/val/lookback{args.lookback}.npz') and \
#                     os.path.exists(f'{args.data_dir}/test/lookback{args.lookback}.npz'):
#                         return
                
#             else:
#                 if os.path.exists(f'{args.data_dir}/train/lookback{args.lookback}_shift_large_train.npz') and \
#                     os.path.exists(f'{args.data_dir}/val/lookback{args.lookback}.npz') and \
#                     os.path.exists(f'{args.data_dir}/test/lookback{args.lookback}.npz'):
#                         return

#         if args.large_train:
#             trajs_large = np.load(os.path.join(args.data_dir, 'large_scale_training.npy'), allow_pickle=True)
#             total_trajs = trajs_large.shape[0]
#             use_traj_num = int(total_trajs * args.large_train_ratio) + 1
#             use_traj_idx = np.random.choice(total_trajs, use_traj_num, replace=False)
#             trajs_large = trajs_large[use_traj_idx]
#         else:
#             trajs_large = None

#         traj_all_path = os.path.join(args.data_dir, 'origin.npy')
#         trajs_all = np.load(traj_all_path, allow_pickle=True)
#         trajs_train = trajs_all[:, :args.data_training_length]
#         trajs_val = trajs_all[:, args.data_training_length:]
#         trajs_test = trajs_all[:, args.data_training_length:]

#         print('Processing data')
#         process_data(args, trajs_all=trajs_all, trajs_train=trajs_train, trajs_val=trajs_val, trajs_test=trajs_test, trajs_large=trajs_large)