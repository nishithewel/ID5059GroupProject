import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy import stats
import gc


def _get_decimal(data):
    num = 3
    data = int(data * 1000)
    while(data % 10 == 0):
        num = num - 1
        data = data / 10
    if num < 0:
        num = 0
    return num


def split_multivalues_columns(data):
    data[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']
         ] = data['P_emaildomain'].str.split('.', expand=True)
    data[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']
         ] = data['R_emaildomain'].str.split('.', expand=True)

    data['device_name'] = data['DeviceInfo'].str.split('/', expand=True)[0]
    data['device_version'] = data['DeviceInfo'].str.split('/', expand=True)[1]

    data['OS_id_30'] = data['id_30'].str.split(' ', expand=True)[0]
    data['version_id_30'] = data['id_30'].str.split(' ', expand=True)[1]

    data['browser_id_31'] = data['id_31'].str.split(' ', expand=True)[0]
    data['version_id_31'] = data['id_31'].str.split(' ', expand=True)[1]

    data['screen_width'] = data['id_33'].str.split('x', expand=True)[0]
    data['screen_height'] = data['id_33'].str.split('x', expand=True)[1]

    data['id_34'] = data['id_34'].str.split(':', expand=True)[1]
    data['id_23'] = data['id_23'].str.split(':', expand=True)[1]

    data.drop(['P_emaildomain', 'R_emaildomain',
               'DeviceInfo', 'id_33', 'id_34', 'id_23'])
# this should be replace function
    data.loc[data['device_name'].str.contains(
        'SM', na=False), 'device_name'] = 'Samsung'
    data.loc[data['device_name'].str.contains(
        'SAMSUNG', na=False), 'device_name'] = 'Samsung'
    data.loc[data['device_name'].str.contains(
        'GT-', na=False), 'device_name'] = 'Samsung'
    data.loc[data['device_name'].str.contains(
        'Moto G', na=False), 'device_name'] = 'Motorola'
    data.loc[data['device_name'].str.contains(
        'Moto', na=False), 'device_name'] = 'Motorola'
    data.loc[data['device_name'].str.contains(
        'moto', na=False), 'device_name'] = 'Motorola'
    data.loc[data['device_name'].str.contains(
        'LG-', na=False), 'device_name'] = 'LG'
    data.loc[data['device_name'].str.contains(
        'rv:', na=False), 'device_name'] = 'RV'
    data.loc[data['device_name'].str.contains(
        'HUAWEI', na=False), 'device_name'] = 'Huawei'
    data.loc[data['device_name'].str.contains(
        'ALE-', na=False), 'device_name'] = 'Huawei'
    data.loc[data['device_name'].str.contains(
        '-L', na=False), 'device_name'] = 'Huawei'
    data.loc[data['device_name'].str.contains(
        'Blade', na=False), 'device_name'] = 'ZTE'
    data.loc[data['device_name'].str.contains(
        'BLADE', na=False), 'device_name'] = 'ZTE'
    data.loc[data['device_name'].str.contains(
        'Linux', na=False), 'device_name'] = 'Linux'
    data.loc[data['device_name'].str.contains(
        'XT', na=False), 'device_name'] = 'Sony'
    data.loc[data['device_name'].str.contains(
        'HTC', na=False), 'device_name'] = 'HTC'
    data.loc[data['device_name'].str.contains(
        'ASUS', na=False), 'device_name'] = 'Asus'

    data.loc[data.device_name.isin(data.device_name.value_counts(
    )[data.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    gc.collect()

    return data

# https://www.kaggle.com/code/davidcairuz/feature-engineering-lightgbm


def _aggregation(data):
    # column_a =
    # column_b =

    for col_a in ['TransactionAmt', 'id_02', 'D15']:

        for col_b in ['card1', 'card4', 'addr1']:

            data[f'{col_a}_to_mean_{col_b}'] = data[col_a] / \
                data.groupby([col_b])[col_a].transform('mean')
            data[f'{col_a}_to_std_{col_b}'] = data[col_a] / \
                data.groupby([col_b])[col_a].transform('std')

    return data


def preprocess(df):
    # === multi values colums related ===
    df = split_multivalues_columns(df)

    # === TransactionAmt related ===
    df["TransactionAmt_decimal"] = df["TransactionAmt"].map(_get_decimal)
    df["TransactionAmt_cents"] = df["TransactionAmt"] - \
        np.floor(df["TransactionAmt"])
    df["TransactionAmt_Log"] = np.log(df["TransactionAmt"])

    # === TransactionDT related ===
    df["Transaction_day_of_week"] = np.floor(
        (df["TransactionDT"] / (3600 * 24) - 1) % 7)
    df["Transaction_hour"] = np.floor(df["TransactionDT"] / 3600) % 24

    # === aggregation related ===
    df = _aggregation(df)
    return df
