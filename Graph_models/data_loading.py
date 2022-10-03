import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch

from config import train_kwargs


class TrafficDataset(Dataset):

    def __init__(self, x, y, scaler, device):
        self.device = device

        self.x = self.np2torch(x)
        self.y = self.np2torch(y)
        self.scaler = scaler

        self.nsample = self.x.shape[0]
        self.indices = self.get_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]

        x = self.x[t]

        y = self.y[t]
        # x = self.np2torch(x)
        # y = self.np2torch(y)

        if len(x.shape) < 4:
            x = torch.unsqueeze(x, -1)

        sample = {'x': x, 'y': y}
        return sample

    def np2torch(self, x):
        x = torch.Tensor(x)
        if torch.cuda.is_available():
            x = x.to(self.device)
        return x

    def get_indices(self):
        indices = np.arange(self.nsample)
        return indices


def data_split(dataset, **kwargs):
    dataset_path = '../data/{}.csv'.format(dataset)
    dataset = pd.read_csv(dataset_path, parse_dates=["time"])
    print('Data shape', dataset.shape)

    dataset = dataset.set_index(['time'])
    dataset.head()

    dataset.isnull().sum()
    np.where(np.isnan(dataset))
    dataset.tail()

    # Train-test split

    print("dataset_shape_aftersampling:", dataset.shape)
    total_steps = dataset.shape[0]
    train_size = int(total_steps * 0.7)
    val_size = int(total_steps * 0.1)

    train_df, val_df, test_df = dataset[0:train_size], dataset[train_size:train_size + val_size], \
                                dataset[train_size + val_size:]  # total dataset
    train_df = pd.DataFrame(train_df)
    val_df = pd.DataFrame(val_df)
    test_df = pd.DataFrame(test_df)
    print("train_shape:", train_df.shape)
    train_df.head()

    print("test_shape:", test_df.shape)
    test_df.tail()

    sc = MinMaxScaler(feature_range=(0, 1))
    sc = sc.fit(train_df)

    train_df = sc.transform(train_df)
    val_df = sc.transform(val_df)
    test_df = sc.transform(test_df)

    # print("train_df", train_df)
    # print("test_df", test_df)
    train_df = pd.DataFrame(train_df)
    val_df = pd.DataFrame(val_df)
    test_df = pd.DataFrame(test_df)
    train_df.tail()
    test_df.tail()

    train_df_inverse = sc.inverse_transform(train_df)
    train_df_inverse = pd.DataFrame(train_df_inverse)
    train_df_inverse.tail()

    val_df_inverse = sc.inverse_transform(val_df)
    val_df_inverse = pd.DataFrame(val_df_inverse)
    val_df_inverse.tail()

    test_df_inverse = sc.inverse_transform(test_df)
    test_df_inverse = pd.DataFrame(test_df_inverse)
    test_df_inverse.tail()

    # Converting the time series to samples
    def create_dataset(X, y, in_seq_len=12, out_seq_len=1):
        Xs, ys = [], []
        for i in tqdm(range(len(X) - in_seq_len)):
            v = X.iloc[i:(i + in_seq_len)].to_numpy()
            Xs.append(v)
            ys.append(y.iloc[i + in_seq_len: i + in_seq_len + out_seq_len])
        return np.array(Xs), np.array(ys)

    in_seq_len = kwargs['in_seq_len']
    out_seq_len = kwargs['out_seq_len']
    x_train, y_train = create_dataset(train_df, train_df, in_seq_len=in_seq_len, out_seq_len=out_seq_len)
    x_val, y_val = create_dataset(val_df, val_df, in_seq_len=in_seq_len, out_seq_len=out_seq_len)
    x_test, y_test = create_dataset(test_df, test_df, in_seq_len=in_seq_len, out_seq_len=out_seq_len)

    print("x_train_shape:", x_train.shape)
    # print("x_train:", x_train)
    print("y_train_shape:", y_train.shape)

    # y_train = pd.DataFrame(y_train)
    # y_train.tail()
    # y_train_shape = y_train.shape
    # if len(y_train.shape) > 2:
    #     y_train_inverse_verif = sc.inverse_transform(y_train.reshape(-1, y_train.shape[-1]))
    #     y_train_inverse_verif = y_train_inverse_verif.reshape(y_train_shape)
    # else:
    #     y_train_inverse_verif = sc.inverse_transform(y_train)
    # y_train_inverse_verif = pd.DataFrame(y_train_inverse_verif)
    # y_train_inverse_verif.tail()

    print("x_val_shape:", x_val.shape)
    print("y_val_shape:", y_val.shape)

    # y_val = pd.DataFrame(y_val)
    # y_val.tail()

    # y_val_shape = y_val.shape
    # if len(y_val.shape) > 2:
    #     y_val_inverse_verif = sc.inverse_transform(y_val.reshape(-1, y_val.shape[-1]))
    #     y_val_inverse_verif = y_val_inverse_verif.reshape(y_val_shape)
    # else:
    #     y_val_inverse_verif = sc.inverse_transform(y_val)
    # y_val_inverse_verif = pd.DataFrame(y_val_inverse_verif)
    # y_val_inverse_verif.tail()

    print("x_test_shape:", x_test.shape)
    # print("x_test", x_test)
    print("y_test_shape:", y_test.shape)
    # y_test = pd.DataFrame(y_test)
    # y_test.tail()

    # y_test_shape = y_test.shape
    # if len(y_test.shape) > 2:
    #     y_test_inverse_verif = sc.inverse_transform(y_test.reshape(-1, y_test.shape[-1]))
    #     y_test_inverse_verif = y_test_inverse_verif.reshape(y_test_shape)
    # else:
    #     y_test_inverse_verif = sc.inverse_transform(y_test)
    # y_test_inverse_verif = pd.DataFrame(y_test_inverse_verif)
    # y_test_inverse_verif.tail()

    print("x_train_shape", x_train.shape)
    print("x_val_shape", x_val.shape)
    print("x_test_shape", x_test.shape)
    print("y_train", y_train.shape)
    print("y_val", y_val.shape)
    print("y_test", y_test.shape)
    print("n_feature", x_train.shape[2])

    return x_train, x_val, x_test, y_train, y_val, y_test, sc


def get_dataloader(dataset, **kwargs):
    x_train, x_val, x_test, y_train, y_val, y_test, scaler = data_split(dataset, **kwargs)

    train_set = TrafficDataset(x_train, y_train, scaler, kwargs['device'])
    val_set = TrafficDataset(x_val, y_val, scaler, kwargs['device'])
    test_set = TrafficDataset(x_test, y_test, scaler, kwargs['device'])

    train_loader = DataLoader(train_set, batch_size=train_kwargs['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=train_kwargs['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=train_kwargs['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, scaler
