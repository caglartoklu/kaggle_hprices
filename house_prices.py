# -*- coding: utf-8 -*-

"""
OneHotEncoder vs get_dummies
"""

#%%
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


#%%
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


#%%
columns = "MSSubClass,MSZoning,LotFrontage,LotArea".split(",")
dftrain = df0[columns].head(10)

# dftest = df2[columns]
dftest = df2[columns].tail(10)
dftest.reset_index(drop=True, inplace=True)

# train için tüm satırları,
# test için sadece son 10 satırı kullandık.
# şu durumu unutmayalım:
# bunu ML'e sokacaksak, kolonları eşit olmalı.
# çünkü, train'e göre fit olacak, test ise predict edilecek.
# diyelim ki bir kategorideki bir değer, sadece train'de var.
# ve test içinde yok.
# bunları ayrı ayrı dummy yapsaydik, veya encode etseydik,
# şu durum olusabilirdi:

# train:
#  red:0 green:1 blue:2 yellow:3
# test:
# red:0 blue:1 yellow:2
# (test verisinde green olmadığını düşünelim.)
# gordugunuz gibi, blue ve yellow, yanlış değerler aldı.

# bu durumda, şunu yapmamız gerekir:
# train içinde hangi encoding'i kullandiysak,
# test'e de onu uygulamaliyiz.

# aşağıdaki kod, bunu ornekliyor.
# MSZoning kolonunu alıyor, train içinde daha fazla value (4 ya da 5) var.
# test icindeyse sadece 2 tane var.



#%%
# LabelEncoder first:
# OneHotEncoder, string alanlar üzerinde çalışmıyor.
# bunun çözmek için, öncelikle LabelEncoder kullanıyoruz.
from sklearn.preprocessing import LabelEncoder
lencoder = LabelEncoder()

column_name = "MSZoning"

# dftrain'deki değerlere göre encoding'i belirlesin:
lencoder.fit(dftrain[column_name])
# değerler, lencoder içinde kalacak.

# aynı değerleri hem dftrain'e, hem dftest'e uygula:
dftrain[column_name + "LEncoded"] = lencoder.transform(dftrain[column_name])
# dftest'de bulunan aynı kolon için aynı encoder kullanmaliyiz:
dftest[column_name + "LEncoded"] = lencoder.transform(dftest[column_name])
# farklı bir kolon için farklı bir encoder kullanmaliydik.

# artık, sayısal bir kolonumuz var.
# dftrain icindeki MSZoningLEncoded kolonuna bakınız. 0..4 değerleri almış olmalı.
# dftest icindeki MSZoningLEncoded kolonuna bakınız. 3 ve 4 değerleri almış olmalı.

# bundan sonra, OneHotEncoder uygulayabiliriz.


#%%
# Then, OneHotEncoder for dftrain
from sklearn.preprocessing import OneHotEncoder
ohencoder = OneHotEncoder()

ohencoder.fit(dftrain[[column_name + "LEncoded"]])

arr_transformed = ohencoder.transform(dftrain[[column_name + "LEncoded"]]).toarray()
df_transformed = pd.DataFrame(arr_transformed)
dftrain = pd.concat([dftrain, df_transformed], axis=1)


#%%
# Then, OneHotEncoder for dftest
# it is already fit.

arr_transformed = ohencoder.transform(dftest[[column_name + "LEncoded"]]).toarray()
df_transformed = pd.DataFrame(arr_transformed)
dftest = pd.concat([dftest, df_transformed], axis=1)


#%%
# hem dftrain hem de dftest içinde 10'ar kolon olduğuna dikkat ediniz.
# dftest içinde 0, 1, 2 diye 3 kolon olacak.
# bunların değerleri tamamen 0.
# neden?
# bu değerleri dolduran hiç bir değer dftest içinde yoktu.
# bunları, OneHotEncoder eklemiş.
# bunlar olmasaydi, dftest, dftrain'den farklı kolonlara sahip olacaktı.
# sonuç olarak da, ML'e giremeyecekti.
# bu sayede, aynı kolonlara sahipler.

# orjinal kolonlar, encode edildikten sonra drop edilmelidir.
# incelenmesi için bıraktım.


#%%
# last
