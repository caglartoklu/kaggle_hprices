# -*- coding: utf-8 -*-

"""
Kaggle House Pricing Dataset
"""

# %%
# import
import os
import pandas as pd
import numpy as np
# import matplotlib as plt
# import seaborn as sns
from sklearn import ensemble
from sklearn import tree
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# %%
# read files

# /kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv
# /kaggle/input/house-prices-advanced-regression-techniques/data_description.txt
# /kaggle/input/house-prices-advanced-regression-techniques/train.csv
# /kaggle/input/house-prices-advanced-regression-techniques/test.csv

if os.path.isdir("/kaggle/input"):
    # we are on Kaggle.
    file_name_train = "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"
    file_name_test = "/kaggle/input/house-prices-advanced-regression-techniques/test.csv"
    file_name_output = "house_prices_submission.csv"
    # import os
    # for dirname, _, filenames in os.walk('/kaggle/input'):
    #     for filename in filenames:
    #         print(os.path.join(dirname, filename))
else:
    # we are not on Kaggle, local machine possibly.
    file_name_train = "train.csv"
    file_name_test = "test.csv"
    file_name_output = "house_prices_submission.csv"

df = pd.read_csv(file_name_train)
df2 = pd.read_csv(file_name_test)
# ids = df2["PassengerId"]
df0 = df.copy()


# %%
# columns = "MSSubClass,MSZoning,LotFrontage,LotArea".split(",")
# dftrain = df0[columns]
# dftest = df2[columns]

# %%
# desc
dfdesc = df.describe().T
dfdesc


# %%
df.info()


# %%
# columns to drop
columns_to_drop = ["Id", "Alley", "FireplaceQu",
                   "PoolQC", "Fence", "MiscFeature"]
# Alley: 91 non-null
# FireplaceQu: 770
# PoolQC: 7
# Fence: 281
# MiscFeature: 54
df = df.drop(columns_to_drop, axis=1)
df2 = df2.drop(columns_to_drop, axis=1)


# %%
dfcategorics = df.select_dtypes(include=['object'])
print(dfcategorics.columns)


#%%
# TODO: burayı elden gecir.
dfi = df.select_dtypes(include=['int64'])
# dff = df.select_dtypes(include=['float64'])

ordinal_columns = """
ExterQual
ExterCond
BsmtQual
BsmtCond
BsmtFinType1
BsmtFinType2
HeatingQC
KitchenQual
GarageFinish
GarageQual
GarageCond
PavedDrive
""".strip().split("\n")

dfo = df[ordinal_columns]
dfx = pd.concat([dfi, dfo], axis=1)

dfx.info()

#%% drop lines with na
# TODO: drop değil fillna uygula.
# SATıRLARı HARCAMA!
dfx = dfx.dropna(axis=0)


#%% encone ordinal columns
def encode_ordinal(df0, column_name:str, categories):
    """
    Label Encodes on a custom order.

    df0: DataFrame
    column_name: str
    categories: List of strings

    Can bu used for ordinal mapping.
    It works on a copy of DataFrame df0.
    It overwrites the column in this copy.

    requires:
        from sklearn.preprocessing import LabelEncoder
    categories: ["red", "green", "blue"]
                  0      1        2

    example:
        df["ExterQual2"] = df["ExterQual"]
        df2 = encode_ordinal(df, "ExterQual2", ["Ex", "Gd", "TA", "Fa", "Po"])

    returns a modified copy of the original DataFrame.
    """
    df = df0.copy()
    mapping = {}
    for i, category in enumerate(categories):
        mapping[category] = i

    df[column_name] = df[column_name].map(mapping)
    return df

values_qualities = ["Ex", "Gd", "TA", "Fa", "Po"]
values_fintype = ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
values_garagefinish = ["NA", "Unf", "RFn", "Fin"]
values_paveddrive = ["N", "P", "Y"]

dfx = encode_ordinal(dfx, "ExterQual", values_qualities)
dfx = encode_ordinal(dfx, "ExterCond", values_qualities)
dfx = encode_ordinal(dfx, "BsmtQual", ["NA"] + values_qualities)
dfx = encode_ordinal(dfx, "BsmtCond", ["NA"] + values_qualities)
dfx = encode_ordinal(dfx, "BsmtFinType1", ["NA"] + values_fintype)
dfx = encode_ordinal(dfx, "BsmtFinType2", ["NA"] + values_fintype)
dfx = encode_ordinal(dfx, "HeatingQC", values_qualities)
dfx = encode_ordinal(dfx, "KitchenQual", values_qualities)
dfx = encode_ordinal(dfx, "GarageFinish", values_garagefinish)
dfx = encode_ordinal(dfx, "GarageQual", ["NA"] + values_qualities)
dfx = encode_ordinal(dfx, "GarageCond", ["NA"] + values_qualities)
dfx = encode_ordinal(dfx, "PavedDrive", values_paveddrive)


# %% info
# df.info()


# %% last
# last
