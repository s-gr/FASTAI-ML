import math
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from utils.my_fastai_utils import *

# We start by processing the raw dataset into a Data Frame
PATH = 'dataset/'
df = pd.read_csv(f'{PATH}Train.csv', low_memory=False, parse_dates=['saledate'])
df.to_pickle('tmp/df_initial.pickle')
df = pd.read_pickle('tmp/df_initial.pickle')

# X = pd.read_pickle('tmp/train_data_X.pickle').values
# y = pd.read_pickle('tmp/train_data_y.pickle').values
# y = y.reshape(-1)
# print(y.shape)


df_raw = df

# Since we are more interested in the ratio than the pure difference, we take the log (rmsle is used to rate the model in the end)
df_raw.SalePrice = np.log(df_raw.SalePrice)

add_datepart(df_raw, 'saledate')
train_cats(df_raw)
print(df_raw.saleYear.head())
df_raw.UsageBand.cat.set_categories(['High', 'Low', 'Medium'], ordered=True, inplace=True)
df_raw.UsageBand = df_raw.UsageBand.cat.codes
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
df, y, nas = proc_df(df_raw, 'SalePrice')


m = RandomForestRegressor(n_jobs=-1)
print(m.fit(X, y))
print(m.score(X, y))


def split_vals(a, n):
    return a[:n].copy(), a[n:].copy()


n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape


def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())


def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):
        res.append(m.oob_score_)
    print(res)
