import os
import warnings

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.tools import StandardScaler

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
                 target='OT'):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.input_len, 12 * 30 * 24 + 4 * 30 * 24 - self.input_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        self.data = self.scaler.transform(df_value)
        self.data_x = self.data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        seq_x = self.data_x[r_begin:r_end]
        if self.set_type == 0:
            per = np.random.permutation(seq_x.shape[1])
            seq_x = seq_x[:, per]
        return seq_x

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1


class Dataset_ETT_min(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv', target='OT'):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.input_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.input_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_value)

        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        seq_x = self.data_x[r_begin:r_end]
        if self.set_type == 0:
            per = np.random.permutation(seq_x.shape[1])
            seq_x = seq_x[:, per]
        return seq_x

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ECL.csv', target='MT_321'):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.data_path = data_path
        self.features = features
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_raw) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_value)

        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        seq_x = self.data_x[r_begin:r_end]
        if self.set_type == 0:
            per = np.random.permutation(seq_x.shape[1])
            seq_x = seq_x[:, per]
        return seq_x

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='train', size=None, freq='Daily', predict=False):
        # size [label_len, pred_len]
        # info
        if size is None:
            self.input_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.input_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.freq = freq
        self.root_path = root_path
        self.predict = predict
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path + 'Dataset/Train/', self.freq + '-train.csv'))
        df_raw.set_index(df_raw.columns[0], inplace=True)
        if self.set_type == 2:
            df_target = pd.read_csv(os.path.join(self.root_path + 'Dataset/Test/',
                                                 self.freq + '-test.csv'))
            df_target.set_index(df_target.columns[0], inplace=True)
            target_data = df_target.values
            df_info = pd.read_csv(os.path.join(self.root_path, 'Dataset/M4-info.csv'))
            df_info.set_index(df_info.columns[0], inplace=True)
            self.info = df_info['SP']
            df_naive = pd.read_csv(os.path.join(self.root_path, 'Point Forecasts/submission-Naive2.csv'))
            df_naive.set_index(df_naive.columns[0], inplace=True)
            naive_data = df_naive.values
            naive_data = naive_data[self.info == self.freq]
        data_x = df_raw.values
        self.data_x = []
        self.target_data = []
        self.naive_data = []
        for index in range(len(data_x)):
            data_current = data_x[index]
            data_current = data_current[~np.isnan(data_current)]
            data_len = data_current.shape[0] - self.input_len - self.pred_len
            if data_len * 0.25 >= 1:
                if self.set_type == 0:  # train
                    self.data_x.append(data_current[0:int(data_len * 0.75) + self.input_len + self.pred_len])
                elif self.set_type == 1:  # val
                    self.data_x.append(data_current[int(data_len * 0.75):])
                else:
                    self.data_x.append(data_current[-self.input_len:])
                    target_current = target_data[index]
                    target_current = target_current[~np.isnan(target_current)]
                    self.target_data.append(target_current)
                    naive_current = naive_data[index]
                    naive_current = naive_current[~np.isnan(naive_current)]
                    self.naive_data.append(naive_current)

    def __getitem__(self, index):
        data_x = self.data_x[index]
        if self.set_type < 2:
            r_begin = np.random.randint(0, len(data_x) - self.input_len - self.pred_len, size=1)[0]
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = data_x[r_begin:r_end]
            return seq_x
        else:
            target_data = self.target_data[index]
            seq_x = np.concatenate([data_x, target_data], axis=-1)
            if self.predict:
                naive_data = self.naive_data[index]
                seq_y = naive_data
                return seq_x, seq_y
            return seq_x

    def __len__(self):
        return len(self.data_x)
