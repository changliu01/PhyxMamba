import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler, RandomSampler
from sklearn.preprocessing import StandardScaler
import glob
import random
import os
import pickle
from tqdm import tqdm
import pandas as pd
from util import *

class TimeSeriesDataset(Dataset):
    def __init__(self, args, mode='train'):

        super().__init__()
        # Search for txt files
        
        folder_name = mode

        if args.exp_name != 'standard':
            folder_name = f'{args.exp_name}_' + folder_name
            
        if mode == 'train':
            if args.large_train:
                data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}_predlen{args.horizon_tr}_large_train_{args.large_train_ratio}.npz')
            else:
                data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}_predlen{args.horizon_tr}.npz')
        else:
            if args.val_long or mode == 'test':
                data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}.npz')
            else:
                data = np.load(f'{args.data_dir}/{folder_name}/short_lookback{args.lookback}_predlen{args.horizon_tr}.npz')
            
        self.X = data['x'] # (sample_num, lookback, channel_num, grids_dim)
        self.Y = data['y'] # (sample_num, horizon, channel_num, grids_dim)

        # Reduce the size of the dataset
        if args.data_ratio < 1.0 and 'train' in folder_name:
            random_indices = np.random.choice(len(self.X), int(len(self.X)*args.data_ratio), replace=False)
            self.X, self.Y = self.X[random_indices], self.Y[random_indices]
        
        # convert to tensor
        self.X = torch.from_numpy(self.X).float().to(args.device)
        self.Y = torch.from_numpy(self.Y).float().to(args.device)
        
        # add Gaussian noise
        if args.noise > 0.0:
            self.X += torch.randn_like(self.X) * args.noise
            self.Y += torch.randn_like(self.Y) * args.noise

        # sampling training data
        if args.final_data_ratio < 1.0 and 'train' in folder_name:
            assert args.data_ratio == 1.0
            random_indices = np.random.choice(len(self.X), int(len(self.X)*args.final_data_ratio), replace=False)
            self.X, self.Y = self.X[random_indices], self.Y[random_indices]

        if args.partial_observe:
            self.X = self.X[:, :, -args.partial_observe_dim:]
            self.Y = self.Y[:, :, -args.partial_observe_dim:]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]
    
class TimeSeriesDatasetFunctionTest(Dataset):
    def __init__(self, args, mode, test_size=None):

        super().__init__()
        # Search for txt files
            # folder_name = mode

        if args.exp_name != 'standard':
            folder_name = 'test'
            folder_name = f'{args.exp_name}_' + folder_name
        else:
            if mode == 'test_large':
                folder_name = 'test'
            elif mode == 'test_ood':
                folder_name = 'test'
        
        if mode == 'test_large':
            if args.system == 'EEG' and args.long_test_length == 600:
                data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}_trajs_num{test_size}_600.npz')
            else:
                assert args.long_test_length == 300
                data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}_trajs_num{test_size}.npz')
        elif mode == 'test_ood':
            data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}_ood_test.npz')
        else:
            raise NotImplementedError
            
        self.X = data['x'] # (sample_num, lookback, channel_num, grids_dim)
        self.Y = data['y'] # (sample_num, horizon, channel_num, grids_dim)

        # convert to tensor
        self.X = torch.from_numpy(self.X).float().to(args.device)
        self.Y = torch.from_numpy(self.Y).float().to(args.device)
        
        # add Gaussian noise
        if args.noise > 0.0:
            self.X += torch.randn_like(self.X) * args.noise
            self.Y += torch.randn_like(self.Y) * args.noise

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]
    
class TimeSeriesDatasetShift(Dataset):

    def __init__(self, args, mode='train'):

        super().__init__()
        # Search for txt files

        folder_name = mode

        if args.exp_name != 'standard':
            folder_name = f'{args.exp_name}_' + folder_name

        if mode == 'test':
            raise NotImplementedError
        
        elif mode == 'train':
            if args.large_train:
                data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}_shift_large_train_{args.large_train_ratio}.npy')
            else:
                data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}_shift.npy')
        else:
            if args.val_long:
                data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}.npz')
            else:
                data = np.load(f'{args.data_dir}/{folder_name}/short_lookback{args.lookback}_shift.npy')

        self.data = data
        # Reduce the size of the dataset
        if args.data_ratio < 1.0 and 'train' in folder_name:
            random_indices = np.random.choice(len(self.data), int(len(self.data)*args.data_ratio), replace=False)
            self.data = self.data[random_indices]
        
        # convert to tensor
        self.data = torch.from_numpy(self.data).float().to(args.device)
        
        # add Gaussian noise
        if args.noise > 0.0:
            self.data += torch.randn_like(self.data) * args.noise

        if args.final_data_ratio < 1.0 and 'train' in folder_name:
            assert args.data_ratio == 1.0
            random_indices = np.random.choice(len(self.data), int(len(self.data)*args.final_data_ratio), replace=False)
            self.data = self.data[random_indices]

        if args.partial_observe:
            self.X = self.X[:, :, -args.partial_observe_dim:]
            self.Y = self.Y[:, :, -args.partial_observe_dim:]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]

class TimeSeriesDatasetMultiSteps(Dataset):

    def __init__(self, args, mode='train'):

        super().__init__()
        # Search for txt files

        folder_name = mode

        if args.exp_name != 'standard':
            folder_name = f'{args.exp_name}_' + folder_name

        if mode == 'train':
            if args.large_train:
                data = np.load(f'{args.data_dir}/{folder_name}/sf_lookback{args.lookback}_predlen{args.horizon_multistep}_large_train_{args.large_train_ratio}.npz')
            else:
                data = np.load(f'{args.data_dir}/{folder_name}/sf_lookback{args.lookback}_predlen{args.horizon_multistep}.npz')
        else:
            if args.val_long or mode == 'test':
                data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}.npz')
            else:
                data = np.load(f'{args.data_dir}/{folder_name}/sfval_short_lookback{args.lookback}_predlen{args.horizon_multistep}.npz')
            
        self.X = data['x'] # (sample_num, lookback, channel_num)
        self.Y = data['y'] # (sample_num, horizon, channel_num)

        self.unified_data = np.concatenate([self.X, self.Y], axis=1)
        self.channel_mean = np.mean(self.unified_data, axis=(0, 1), keepdims=False)
        self.channel_variance = np.var(self.unified_data, axis=(0, 1), keepdims=False)

        # Reduce the size of the dataset
        if args.multi_step_data_ratio < 1.0 and 'train' in folder_name:
            random_indices = np.random.choice(len(self.X), int(len(self.X)*args.multi_step_data_ratio), replace=False)
            self.X, self.Y = self.X[random_indices], self.Y[random_indices]
        
        # convert to tensor
        self.X = torch.from_numpy(self.X).float().to(args.device)
        self.Y = torch.from_numpy(self.Y).float().to(args.device)
        
        # add Gaussian noise
        if args.noise > 0.0:
            self.X += torch.randn_like(self.X) * args.noise
            self.Y += torch.randn_like(self.Y) * args.noise

        if args.final_data_ratio < 1.0 and 'train' in folder_name:
            assert args.multi_step_data_ratio == 1.0
            random_indices = np.random.choice(len(self.X), int(len(self.X)*args.final_data_ratio), replace=False)
            self.X, self.Y = self.X[random_indices], self.Y[random_indices]

        if args.partial_observe:
            self.X = self.X[:, :, -args.partial_observe_dim:]
            self.Y = self.Y[:, :, -args.partial_observe_dim:]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]
    
class TimeSeriesDatasetHopfield(Dataset):

    def __init__(self, args, mode='train'):

        super().__init__()
        # Search for txt files
        if mode == 'test_bench':
            folder_name = 'test'
        else:
            folder_name = mode

        if args.exp_name != 'standard':
            folder_name = f'{args.exp_name}_' + folder_name

        if mode == 'train':
            if args.large_train:
                data = np.load(f'{args.data_dir}/{folder_name}/sf_lookback{args.lookback}_predlen{args.horizon_multistep}_large_train_{args.large_train_ratio}.npz')
            else:
                data = np.load(f'{args.data_dir}/{folder_name}/sf_lookback{args.lookback}_predlen{args.horizon_multistep}.npz')
                
        elif mode == 'test_bench':
            data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}_test_bench.npz')

        else:
            if args.val_long or mode == 'test':
                data = np.load(f'{args.data_dir}/{folder_name}/lookback{args.lookback}.npz')
            else:
                data = np.load(f'{args.data_dir}/{folder_name}/sfval_short_lookback{args.lookback}_predlen{args.horizon_multistep}.npz')
            
        self.X = data['x'] # (sample_num, lookback, channel_num, grids_dim)
        self.Y = data['y'] # (sample_num, horizon, channel_num, grids_dim)

        # Reduce the size of the dataset
        if args.multi_step_data_ratio < 1.0 and 'train' in folder_name:
            random_indices = np.random.choice(len(self.X), int(len(self.X)*args.data_ratio), replace=False)
            self.X, self.Y = self.X[random_indices], self.Y[random_indices]
        
        # convert to tensor
        self.X = torch.from_numpy(self.X).float().to(args.device)
        self.Y = torch.from_numpy(self.Y).float().to(args.device)
        
        # add Gaussian noise
        if args.noise > 0.0:
            self.X += torch.randn_like(self.X) * args.noise
            self.Y += torch.randn_like(self.Y) * args.noise

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]


class SingleSubjectDataset(Dataset):
    def __init__(self, data, seq_len):
        """
        参数:
            data: 形状为 (num_timesteps, num_features) 的张量，单个被试的时间序列数据
            seq_len: 输入序列的长度
            device: 数据存储设备，例如 "cpu" 或 "cuda"
        """
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.num_samples = max(0, len(data) - seq_len + 1)  # 可生成的窗口数量，避免负值

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 返回一个时间窗口，形状为 (seq_len, num_features)
        return self.data[idx:idx + self.seq_len]

class ContrastiveMultiSubjectDataset(Dataset):

    def __init__(self, args, mode):

        super().__init__()

        assert args.exp_name in ['contrastive_context_in', 'contrastive_context_out', 'contrastive_context_full']
        
        data_path = os.path.join(args.data_dir, args.exp_name, f'{mode}.npy')
        self.data = np.load(data_path)
        self.data = torch.from_numpy(self.data).float()

        self.batch_per_epoch = args.contrastive_batch_per_epoch
        self.batch_size = args.contrastive_batch_size

        if self.data.ndim == 2:  # 如果是单个被试，增加一个维度
            self.data = self.data.unsqueeze(0)
        
        self.num_subjects = self.data.shape[0]

        self.datasets = [SingleSubjectDataset(self.data[i], args.lookback) for i in range(self.num_subjects)]

        if mode == "train" and args.contrastive_noise_level > 0.0:
            for ds in self.datasets:
                ds.data += torch.randn_like(ds.data) * args.contrastive_noise_level

        total_samples = self.batch_per_epoch * self.batch_size
        num_repeats = (total_samples + len(self.datasets) - 1) // len(self.datasets)  # 向上取整
        self.infinite_data_idx = torch.cat([torch.randperm(len(self.datasets)) for _ in range(num_repeats)])[:total_samples]

        # self.infinite_data_idx = torch.cat([torch.randperm(len(self.datasets)) for _ in range((self.batch_per_epoch * self.batch_size) // len(self.datasets))])

    def __len__(self):

        return self.batch_per_epoch * self.batch_size
    
    def __getitem__(self, idx):
        
        here_idx = self.infinite_data_idx[idx].item()

        idx1 = np.random.randint(0, len(self.datasets[here_idx]))
        idx2 = np.random.randint(0, len(self.datasets[here_idx]))

        return self.datasets[here_idx][idx1], self.datasets[here_idx][idx2], here_idx

class Dataset_Custom_Shift(Dataset):

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        # return seq_x, seq_y, seq_x_mark, seq_y_mark
        return seq_x

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_Custom_MultiSteps(Dataset):

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
    

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y
        # return seq_x

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
        # return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)