# -*- coding: utf-8 -*-

"""
Kaggle House Pricing Dataset
"""

# %%
# import
from pandas.core.algorithms import value_counts
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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


# TODO: fillna


# %%
dfnulls = df2.isnull().sum()
dfnulls


# %% value_counts
def display_value_counts(df1):
    column_names = """
    BsmtQual         44
    BsmtCond         45
    BsmtFinType1     42
    BsmtFinType2     42
    KitchenQual       1
    GarageFinish     78
    GarageQual       78
    GarageCond       78
    """.strip()

    for line in column_names.splitlines():
        column_name = line.split()[0]
        print()
        print(column_name)
        value_counts = df1[column_name].value_counts()
        print(value_counts)

# display_value_counts(df)
# display_value_counts(df2)


# %% fillna
# fillna
def fillna(df1):
    """
    Dışarıdan gelen DataFrame'i degistirir.
    Bu sebeple ayrı copy() işlerine girmedik.
    """
    # TODO: değerleri korelasyona bakarak doldur.
    df1["BsmtQual"] = df1["BsmtQual"].fillna("TA")
    df1["BsmtCond"] = df1["BsmtCond"].fillna("TA")
    df1["BsmtFinType1"] = df1["BsmtFinType1"].fillna("GLQ")
    df1["BsmtFinType2"] = df1["BsmtFinType2"].fillna("Unf")
    df1["KitchenQual"] = df1["KitchenQual"].fillna("TA")
    df1["GarageFinish"] = df1["GarageFinish"].fillna("Unf")
    df1["GarageQual"] = df1["GarageQual"].fillna("TA")
    df1["GarageCond"] = df1["GarageCond"].fillna("TA")

    # TODO: mean ile doldurduk, daha düzgün dolduralim:
    # TODO: mean yapiyorsan, bari int olanları int bırak:
    df1["LotFrontage"] = df1["LotFrontage"].fillna(df1["LotFrontage"].mean())
    df1["MasVnrArea"] = df1["MasVnrArea"].fillna(df1["MasVnrArea"].mean())
    df1["GarageYrBlt"] = df1["GarageYrBlt"].fillna(df1["GarageYrBlt"].mean())

    df1["BsmtFinSF1"] = df1["BsmtFinSF1"].fillna(df1["BsmtFinSF1"].mean())
    df1["BsmtFinSF2"] = df1["BsmtFinSF2"].fillna(df1["BsmtFinSF2"].mean())
    df1["BsmtUnfSF"] = df1["BsmtUnfSF"].fillna(df1["BsmtUnfSF"].mean())
    df1["TotalBsmtSF"] = df1["TotalBsmtSF"].fillna(df1["TotalBsmtSF"].mean())
    df1["BsmtFullBath"] = df1["BsmtFullBath"].fillna(
        df1["BsmtFullBath"].mean())
    df1["BsmtHalfBath"] = df1["BsmtHalfBath"].fillna(
        df1["BsmtHalfBath"].mean())
    df1["GarageCars"] = df1["GarageCars"].fillna(df1["GarageCars"].mean())
    df1["GarageArea"] = df1["GarageArea"].fillna(df1["GarageArea"].mean())


fillna(df)
fillna(df2)


# %%
def build_df(df1):
    df = df1.copy()
    dfi = df.select_dtypes(include=['int64'])
    dff = df.select_dtypes(include=['float64'])
    # TODO: ordinal olmayan kategorik kolonları da hallet.

    # önce int64 sonra float64 kolonları secmek,
    # kolonlardaki na durumuna göre, farklı kolon siralari ortaya çıkabilir.
    # bunu halledelim.

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
    dfx = pd.concat([dfi, dfo, dff], axis=1)
    return dfx


df = build_df(df)
df2 = build_df(df2)


# %% encone ordinal columns
def encode_ordinal(df0, column_name: str, categories):
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


def encode_ordinal_columns(df0):
    df1 = df0.copy()
    values_qualities = ["Ex", "Gd", "TA", "Fa", "Po"]
    values_qualities_with_na = ["NA", "Ex", "Gd", "TA", "Fa", "Po"]
    values_fintype = ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
    values_garagefinish = ["NA", "Unf", "RFn", "Fin"]
    values_paveddrive = ["N", "P", "Y"]

    df1 = encode_ordinal(df1, "ExterQual", values_qualities)
    df1 = encode_ordinal(df1, "ExterCond", values_qualities)
    df1 = encode_ordinal(df1, "BsmtQual", values_qualities_with_na)
    df1 = encode_ordinal(df1, "BsmtCond", values_qualities_with_na)
    df1 = encode_ordinal(df1, "BsmtFinType1", values_fintype)
    df1 = encode_ordinal(df1, "BsmtFinType2", values_fintype)
    df1 = encode_ordinal(df1, "HeatingQC", values_qualities)
    df1 = encode_ordinal(df1, "KitchenQual", values_qualities)
    df1 = encode_ordinal(df1, "GarageFinish", values_garagefinish)
    df1 = encode_ordinal(df1, "GarageQual", values_qualities_with_na)
    df1 = encode_ordinal(df1, "GarageCond", values_qualities_with_na)
    df1 = encode_ordinal(df1, "PavedDrive", values_paveddrive)
    return df1


df = encode_ordinal_columns(df)
df2 = encode_ordinal_columns(df2)


# %%
dfnulls1 = pd.isnull(df).sum()
dfnulls2 = pd.isnull(df2).sum()


# %%
# TODO: diğer categoric alanları da OneHotEncoder'a sok.


# %%
# split
target = "SalePrice"
x = df.drop(target, axis=1)
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                    random_state=18)


# %%
# ML

models = [LinearRegression(), RandomForestRegressor(random_state=23),
          GradientBoostingRegressor(n_estimators=500, random_state=23)]

for model in models:
    model.fit(x_train, y_train)
    print("_" * 20)
    print(str(model))
    score = model.score(x_test, y_test)
    print(score)


# %% info
# df.info()


# %%
# extract output
# predictions = model.predict(df2)
# my_dict = {"Id": test_data.Id, "SalePrice": predictions}
# output = pd.DataFrame(my_dict)
# output.to_csv('sample_submission.csv', index=False)


# %% last
# last
