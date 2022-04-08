import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy import stats
import gc

from sklearn.preprocessing import OneHotEncoder


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

    # data['id_34'] = data['id_34'].str.split(':', expand=True)[1]
    # data['id_23'] = data['id_23'].str.split(':', expand=True)[1]

    # data.drop(['P_emaildomain', 'R_emaildomain',
    #            'DeviceInfo', 'id_33', 'id_34', 'id_23'])
# this should be replace function


# dict(SM='Samsung',SAMSUNG)
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


def convert_to_cats(df):
    """"GIven athe merged dataframe, this function will
    conver the datatype of categorical columns to 
    data of type category"""
    obj_features = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_19', 'id_20',
                    'id_28', 'id_29',
                    'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
                    'DeviceType', 'DeviceInfo', 'ProductCD',
                    'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                    'P_emaildomain', 'R_emaildomain',
                    'addr1', 'addr2',
                    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
    df[obj_features] = df[obj_features].apply(lambda x: x.astype('category'))
    return df


def get_dummy_vars(test, train, dummy_cols):

    for col in dummy_cols:
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        dummy_train = enc.fit_transform(train[col].to_numpy().reshape(-1, 1))
        new_col = [col+'_' + str(i) for i in enc.categories_[0].tolist()]

        # for df in [train, test]:
        dummy_test = enc.transform(test[col].to_numpy().reshape(-1, 1))

        dummy_dftest = pd.DataFrame(dummy_test, columns=new_col)
        dummy_dftrain = pd.DataFrame(dummy_train, columns=new_col)

        test = pd.concat([test, dummy_dftest], axis=1)
        train = pd.concat([train, dummy_dftrain], axis=1)

    return test, train


def preprocess(df):
    # === multi values colums related ===
    df = convert_to_cats(df)
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
