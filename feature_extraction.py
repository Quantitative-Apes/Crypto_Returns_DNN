import pandas as pd
import math
import torch
# 'open_t',
# 'open',
# 'high',
# 'low',
# 'close',
# 'volume',
# 'close_t',
# 'qav', # quote asset volume
# 'n_trades',
# 'tbbav', # taker buy base asset volume,
# 'tbqav', # taker buy quote asset volume,
# 'ignore',

def feature_dfs_from_klines(path:str, train_frac=0.9):
    """Load klines, convert into features, split into train and test data"""
    df = pd.read_csv(path)

    price_df = df[['open', 'high', 'low', 'close']]
    price_df = (price_df - price_df.shift(1))/price_df.shift(1)
    price_df['volume'] = df['volume']
    price_df['qav'] = df['qav']
    price_df['n_trades'] = df['n_trades']
    price_df['tbbav'] = df['tbbav']
    price_df['tbqav'] = df['tbqav']
    price_df['next_return'] = price_df['close'].shift(-1)
    price_df = price_df.dropna()

    last_train_idx = math.floor(len(price_df)*train_frac)
    train = price_df[:last_train_idx].copy()
    if train_frac != 1:
        test = price_df[last_train_idx+1:].copy()
    else:
        test = None
    dists = {}
    
    minimum = train.min()
    maximum = train.max()
    train = -1 + (train - minimum)*2/(maximum-minimum)
    test = -1 + (test - minimum)*2/(maximum-minimum)
    return train, test, (minimum, maximum)

def rescale_df(df, minimum, maximum):
    df = (df+1)/2 * (maximum-minimum) + minimum
    return df

class SequenceDataset(torch.utils.data.Dataset):
    """Dataset class that provides sequences of data from a pandas dataframe"""
    def __init__(self, df, target, features, sequence_length=30):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.X = torch.tensor(df[features].values).float()
        self.y = torch.tensor(df[target].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        # courtesy of https://www.crosstab.io/articles/time-series-pytorch-lstm
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]