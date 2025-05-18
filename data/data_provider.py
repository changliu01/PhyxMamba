import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import TimeSeriesDataset, TimeSeriesDatasetShift, TimeSeriesDatasetMultiSteps, TimeSeriesDatasetFunctionTest, TimeSeriesDatasetHopfield, ContrastiveMultiSubjectDataset, Dataset_Custom_Shift, Dataset_Custom, Dataset_Custom_MultiSteps
import networkx as nx
import random
import pickle
import logging

def data_provider(args, mode, test_size=None):
    
    if args.data_type == 'real':

        if mode == 'train' or mode == 'val':
            
            if args.shift_dataset:
                dataset = Dataset_Custom_Shift(args, root_path=f'./data/{args.system}', flag='train', 
                                               size=[args.lookback, 0, 0], features='M', data_path=f'{args.system}.csv', target='OT', scale=True, timeenc=1)
            else:
                raise ValueError('Not implemented')
        elif mode == 'train_stage1':
            dataset = Dataset_Custom_Shift(args, root_path=f'./data/{args.system}', flag='train', 
                                               size=[args.lookback, 0, 0], features='M', data_path=f'{args.system}.csv', target='OT', scale=True, timeenc=1)
        elif mode == 'train_stage2':
            dataset = Dataset_Custom_MultiSteps(args, root_path=f'./data/{args.system}', flag='train',
                                               size=[args.lookback, 0, args.horizon_multistep], features='M', data_path=f'{args.system}.csv', target='OT', scale=True, timeenc=1)
        elif mode == 'val_stage1':  
            if args.shift_dataset:
                dataset = Dataset_Custom_Shift(args, root_path=f'./data/{args.system}', flag='val', 
                                               size=[args.lookback, 0, 0], features='M', data_path=f'{args.system}.csv', target='OT', scale=True, timeenc=1)
            else:
                raise ValueError('Not implemented')
            
        elif mode == 'val_stage2':
            dataset = Dataset_Custom_MultiSteps(args, root_path=f'./data/{args.system}', flag='val',
                                               size=[args.lookback, 0, args.horizon_multistep], features='M', data_path=f'{args.system}.csv', target='OT', scale=True, timeenc=1)
            
        elif mode == 'test':
            dataset = Dataset_Custom_MultiSteps(args, root_path=f'./data/{args.system}', flag='test',
                                               size=[args.lookback, 0, args.horizon_multistep], features='M', data_path=f'{args.system}.csv', target='OT', scale=True, timeenc=1)
            
        else:
            raise ValueError('Not implemented')


    elif args.data_type == 'sim':
        if mode == 'train' or mode == 'val':
            if args.shift_dataset:
                dataset = TimeSeriesDatasetShift(args, mode)
            else:
                dataset = TimeSeriesDataset(args, mode)
        elif mode == 'train_stage1':
            dataset = TimeSeriesDatasetShift(args, 'train')
        elif mode == 'train_stage2':
            dataset = TimeSeriesDatasetMultiSteps(args, 'train')
        elif mode == 'val_stage1':
            if args.shift_dataset:
                dataset = TimeSeriesDatasetShift(args, 'val')
            else:
                dataset = TimeSeriesDataset(args, 'val')
        elif mode == 'val_stage2':
            dataset = TimeSeriesDatasetMultiSteps(args, 'val')
        elif mode == 'test_large':
            dataset = TimeSeriesDatasetFunctionTest(args, mode, test_size)
        elif mode == 'test_ood':
            dataset = TimeSeriesDatasetFunctionTest(args, mode)

        else:
            dataset = TimeSeriesDataset(args, mode)

    # if args.shift_dataset:
    #     dataset = TimeSeriesDatasetShift(args, mode)
    # else:
    #     dataset = TimeSeriesDataset(args, mode)
        
    print(mode, len(dataset))
    
    logging.info(f'{mode} data length: {len(dataset)}')

    data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True if mode == 'train' else False,
            num_workers=0,
            drop_last=False)
    
    return dataset, data_loader

def data_provider_encoder_only(args, mode, test_size=None):
    
    # if mode == 'train':
    #     dataset = TimeSeriesDatasetMultiSteps(args, 'train')
    # ######

    if mode == 'train' or mode == 'val':
        dataset = TimeSeriesDatasetMultiSteps(args, mode)

    # if mode == 'train' or mode == 'val':
    #     if args.shift_dataset:
    #         dataset = TimeSeriesDatasetShift(args, mode)
    #     else:
    #         dataset = TimeSeriesDataset(args, mode)
    # elif mode == 'train_stage1':
    #     dataset = TimeSeriesDatasetShift(args, 'train')
    # elif mode == 'train_stage2':
    #     dataset = TimeSeriesDatasetMultiSteps(args, 'train')
    # elif mode == 'val_stage1':
    #     if args.shift_dataset:
    #         dataset = TimeSeriesDatasetShift(args, 'val')
    #     else:
    #         dataset = TimeSeriesDataset(args, 'val')
    # elif mode == 'val_stage2':
    #     dataset = TimeSeriesDatasetMultiSteps(args, 'val')
    elif mode == 'test_large':
        dataset = TimeSeriesDatasetFunctionTest(args, mode, test_size)
    elif mode == 'test_ood':
        dataset = TimeSeriesDatasetFunctionTest(args, mode)

    else:
        dataset = TimeSeriesDataset(args, mode)

    # if args.shift_dataset:
    #     dataset = TimeSeriesDatasetShift(args, mode)
    # else:
    #     dataset = TimeSeriesDataset(args, mode)
        
    print(mode, len(dataset))
    
    logging.info(f'{mode} data length: {len(dataset)}')

    data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True if mode == 'train' else False,
            num_workers=0,
            drop_last=False)
    
    return dataset, data_loader


def data_provider_context(args, mode):

    dataset = ContrastiveMultiSubjectDataset(args, mode)
    
    # print(mode, len(dataset))
    # logging.info(f'{mode} data length: {len(dataset)}')
    data_loader = DataLoader(
            dataset,
            batch_size=args.contrastive_batch_size,
            shuffle=False if mode == 'train' else False,
            num_workers=0,
            drop_last=False)
    
    return dataset, data_loader


def data_provider_hopfield(args, mode, test_size=None):
    
    if mode == 'train_stage1' or mode == 'train_stage2':
        dataset = TimeSeriesDatasetHopfield(args, 'train')
    elif mode == 'val_stage1' or mode == 'val_stage2':
        dataset = TimeSeriesDatasetHopfield(args, 'val')
    elif mode == 'test':
        dataset = TimeSeriesDatasetHopfield(args, 'test')
    elif mode == 'test_large':
        dataset = TimeSeriesDatasetFunctionTest(args, mode, test_size)
    elif mode == 'test_ood':
        dataset = TimeSeriesDatasetFunctionTest(args, mode)
    elif mode == 'test_bench':
        dataset = TimeSeriesDatasetHopfield(args, 'test_bench')

    else:
        raise NotImplementedError
    
    print(mode, len(dataset))
    logging.info(f'{mode} data length: {len(dataset)}')

    data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True if mode == 'train' else False,
            num_workers=0,
            drop_last=False)
    
    return dataset, data_loader


def data_provider_rc(args, mode, is_tune=False):

    # if mode == 'train':
    #     if is_tune:
    #         data = np.load(f'{args.data_dir}/origin_tune_{mode}.npy')
    #     else:
    #         data = np.load(f'{args.data_dir}/origin_{mode}.npy')
    # elif mode == 'test' or mode == 'val':
    #     if is_tune:
    #         data = np.load(f'{args.data_dir}/origin_tune.npy')
    #     else:
    #         data = np.load(f'{args.data_dir}/origin.npy')

    # else:
    #     raise NotImplementedError


    if is_tune:
        raise NotImplementedError
    else:
        if mode == 'train':
            trajs_all = np.load(f'{args.data_dir}/long_traj.npy')
            data = trajs_all[:, -1000:, :]
            data = data[:, :args.data_training_length, :]
        elif mode == 'val':
            trajs_all = np.load(f'{args.data_dir}/long_traj.npy')
            data = trajs_all[:, -1000:, :]
            data = data[:, args.data_training_length:args.data_training_length+args.data_valid_length, :]
        elif mode == 'test':
            trajs_all = np.load(f'{args.data_dir}/long_traj.npy')
            data = trajs_all[:, -1000:, :]
        else:
            raise NotImplementedError

    # if is_tune:
    #     data = np.load(f'{args.data_dir}/origin_tune.npy')
    # else:
    #     data = np.load(f'{args.data_dir}/origin.npy')
    
    edge_info = {}
        
    if args.model == 'PRC' or args.model == 'HoGRC':

        if not os.path.exists(f'{args.data_dir}/edge_info.pkl'):
            if args.direc:
                net = nx.DiGraph() 
            else:
                net = nx.Graph()
            net.add_nodes_from(np.arange(args.node_num))
            if args.net_nam == 'er': 
                for u, v in nx.erdos_renyi_graph(args.node_num, 0.6).edges():
                    net.add_edge(u,v,weight=random.uniform(0,1))
            elif args.net_nam == 'ba':
                for u, v in nx.barabasi_albert_graph(args.node_num, 4).edges():
                    net.add_edge(u,v,weight=random.uniform(1,1)) 
            elif args.net_nam == 'rg':
                for u, v in nx.random_regular_graph(4,args.node_num).edges():
                    net.add_edge(u,v,weight=random.uniform(1,1))
            elif args.net_nam == 'edges':
                edges = np.array([[0,1],[0,2],[0,3],[0,4],[1,0],[1,2],[1,3],[2,1],[2,4],[3,2]])
                weights = np.array([0.4,0.3,0.2,0.1,0.2,0.3,0.5,0.3,0.7,1.0])
                for i in range(edges.shape[0]):
                    net.add_edge(edges[i,0],edges[i,1],weight=weights[i])
                print(args.net_nam)

            else:
                raise NotImplementedError
            
            for i in range(args.node_num):
                swei = 0
                for j in net.neighbors(i):
                    swei = swei+net.get_edge_data(i,j)['weight'] 
                for j in net.neighbors(i):
                    net.edges[i,j]['weight'] = net.edges[i,j]['weight']/swei

            edge_info['edges_out'] = net.edges()
            weights = []
            for u,v in net.edges():
                weights.append(net.edges[u,v]['weight'])
            edge_info['weights'] = weights
            if args.system == 'Lorenz' or 'Lorenz_RC':
                edge_info['edges_in'] = np.array([[1,1],[1,2],[2,2],[2,5],[4,4],[4,3]])
                edge_info['edges_ex'] = np.array([[1],[2],[2]])
            elif args.system == 'Rossler':
                edge_info['edges_in'] = np.array([[1,2],[1,4],[2,1],[2,2],[4,5]])
                edge_info['edges_ex'] = np.array([[1],[1],[1]])
            elif args.system == 'Lorenz96':
                edge_info['edges_in'] = np.array([[1,1],[1,18],[1,24],
                                          [2,2],[2,5],[2,17],
                                          [4,4],[4,10],[4,3],
                                          [8,8],[8,20],[8,6],
                                          [16,16],[16,9],[16,12]])
                edge_info['edges_ex'] = np.array([[1],[-1],[2]])
            else:
                NotImplementedError

            pickle.dump(edge_info, open(f'{args.data_dir}/edge_info.pkl', 'wb'))

        else:
            edge_info = pickle.load(open(f'{args.data_dir}/edge_info.pkl', 'rb'))

    return data, edge_info


def data_provider_esn(args, mode, test_size=None, is_tune=False):
    
    if is_tune:
        # data = np.load(f'{args.data_dir}/origin_tune.npy')
        trajs_all = np.load(f'{args.data_dir}/long_traj_tune.npy')
        # data = trajs_all[:, :-1000, :]
        data = trajs_all

    else:
        # if args.large_train:
        #     if mode == 'train':
        #         data = np.load(f'{args.data_dir}/large_scale_training.npy')
        #         total_traj_num = data.shape[0]
        #         use_traj_num = round(total_traj_num * args.large_train_ratio)
        #         # use_traj_num = int(total_traj_num * args.large_train_ratio)
        #         data = data[:use_traj_num]
        #     elif mode == 'val':
        #         data = np.load(f'{args.data_dir}/large_scale_val.npy')
        #         total_traj_num = data.shape[0]
        #         use_traj_num = round(total_traj_num * args.validation_ratio)
        #         # use_traj_num = int(total_traj_num * args.validation_ratio)
        #         data = data[:use_traj_num]

        #     else:
        #         data = np.load(f'{args.data_dir}/origin.npy')
        # else:
        # data = np.load(f'{args.data_dir}/origin.npy')

        if 'large' in mode:
            
            # if args.system == 'Lorenz' or args.system == 'Rossler' or args.system == 'Lorenz96':
            #     multi_size = 100
            # elif args.system == 'ECG':
            #     multi_size = 10
            # elif args.system == 'EEG':
            #     multi_size = 8
            # else:
            #     raise NotImplementedError
            
            
            if args.exp_name != 'standard':
                folder_name = 'test'
                folder_name = f'{args.exp_name}_' + folder_name
            else:
                if mode == 'test_large':
                    folder_name = 'test'
                elif mode == 'full_large':
                    folder_name = 'test'
            
            if args.system == 'EEG':
                ori_data = np.load(f'{args.data_dir}/{folder_name}/lookback30_trajs_num{test_size}_600.npz', allow_pickle=True)
            else:
                ori_data = np.load(f'{args.data_dir}/{folder_name}/lookback30_trajs_num{test_size}.npz', allow_pickle=True)

            if mode == 'full_large':
                trajs_all = ori_data['x']
            elif mode == 'test_large':
                trajs_all = ori_data['y']
            else:
                raise NotImplementedError
            
            data = trajs_all

            # trajs_all = np.load(f'{args.data_dir}/mult_initial_{multi_size}.npy')
            # data = trajs_all
        
        else:

            trajs_all = np.load(f'{args.data_dir}/long_traj.npy')
            # data = trajs_all[:, -1000:, :]
            data = trajs_all

    
    edge_info = {}

    return data, edge_info