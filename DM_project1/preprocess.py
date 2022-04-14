import matplotlib.pyplot as plt
import matplotlib
from io import StringIO
from sklearn import datasets
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
# %matplotlib inline

# 印出所有的行，列
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 讀檔
train_data = pd.read_csv("tr.csv", dtype=object)
# outcome移到最後
feature = train_data.columns.values
feature = np.append(feature[1:], [feature[0]])
train_data = train_data[feature]

#plt.scatter(train_data["AGE"], train_data["OP_time_minute"])
# plt.show()

# (1) data clearning
# 將性別F,M用0與1替代
train_data[['SEX']] = train_data[['SEX']].replace(['F', 'M'], ['0', '1'])
# 將Joint的TKA,THA用0與1替代
train_data[['Joint']] = train_data[['Joint']
                                   ].replace(['TKA', 'THA'], ['0', '1'])
train_data.fillna(np.nan, inplace=True)


train_data[['AGE', 'LOS', 'OP_time_minute', 'CBC_WBC', 'CBC_RBC', 'CBC_HG', 'CBC_HT', 'CBC_MCV', 'CBC_MCH', 'CBC_MCHC', 'CBC_RDW', 'CBC_Platelet', 'CBC_RDWCV', 'BUN', 'Crea', 'GOT', 'GPT', 'ALB', 'Na', 'K', 'UA']] = train_data[[
    'AGE', 'LOS', 'OP_time_minute', 'CBC_WBC', 'CBC_RBC', 'CBC_HG', 'CBC_HT', 'CBC_MCV', 'CBC_MCH', 'CBC_MCHC', 'CBC_RDW', 'CBC_Platelet', 'CBC_RDWCV', 'BUN', 'Crea', 'GOT', 'GPT', 'ALB', 'Na', 'K', 'UA']].astype(float)

train_data['AGE'] = pd.cut(x=train_data['AGE'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                           labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
train_data['AGE'].fillna(np.nan, inplace=True)
# print(train_data)

# 處理異常值 ， 四分位距
'''
a = train_data[["OP_time_minute", 'CBC_WBC', 'CBC_RBC', 'CBC_HG', 'CBC_HT', 'CBC_MCV', 'CBC_MCH', 'CBC_MCHC',
                'CBC_RDW', 'CBC_Platelet', 'CBC_RDWCV', 'BUN', 'Crea', 'GOT', 'GPT', 'ALB', 'Na', 'K', 'UA']].quantile(0.75)
b = train_data[["OP_time_minute", 'CBC_WBC', 'CBC_RBC', 'CBC_HG', 'CBC_HT', 'CBC_MCV', 'CBC_MCH', 'CBC_MCHC',
                'CBC_RDW', 'CBC_Platelet', 'CBC_RDWCV', 'BUN', 'Crea', 'GOT', 'GPT', 'ALB', 'Na', 'K', 'UA']].quantile(0.25)
c = train_data[["OP_time_minute", 'CBC_WBC', 'CBC_RBC', 'CBC_HG', 'CBC_HT', 'CBC_MCV', 'CBC_MCH',
                'CBC_MCHC', 'CBC_RDW', 'CBC_Platelet', 'CBC_RDWCV', 'BUN', 'Crea', 'GOT', 'GPT', 'ALB', 'Na', 'K', 'UA']]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

#train_data.dropna(axis=1, how='any', inplace=True)
'''


a = train_data["OP_time_minute"].quantile(0.75)
b = train_data["OP_time_minute"].quantile(0.25)
c = train_data["OP_time_minute"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)
a = train_data["CBC_WBC"].quantile(0.75)
b = train_data["CBC_WBC"].quantile(0.25)
c = train_data["CBC_WBC"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)
a = train_data["CBC_RBC"].quantile(0.75)
b = train_data["CBC_RBC"].quantile(0.25)
c = train_data["CBC_RBC"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["CBC_HG"].quantile(0.75)
b = train_data["CBC_HG"].quantile(0.25)
c = train_data["CBC_HG"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["CBC_HT"].quantile(0.75)
b = train_data["CBC_HT"].quantile(0.25)
c = train_data["CBC_HT"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["CBC_MCV"].quantile(0.75)
b = train_data["CBC_MCV"].quantile(0.25)
c = train_data["CBC_MCV"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["CBC_MCH"].quantile(0.75)
b = train_data["CBC_MCH"].quantile(0.25)
c = train_data["CBC_MCH"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["CBC_MCHC"].quantile(0.75)
b = train_data["CBC_MCHC"].quantile(0.25)
c = train_data["CBC_MCHC"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["CBC_RDW"].quantile(0.75)
b = train_data["CBC_RDW"].quantile(0.25)
c = train_data["CBC_RDW"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["CBC_Platelet"].quantile(0.75)
b = train_data["CBC_Platelet"].quantile(0.25)
c = train_data["CBC_Platelet"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["CBC_RDWCV"].quantile(0.75)
b = train_data["CBC_RDWCV"].quantile(0.25)
c = train_data["CBC_RDWCV"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["BUN"].quantile(0.75)
b = train_data["BUN"].quantile(0.25)
c = train_data["BUN"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["Crea"].quantile(0.75)
b = train_data["Crea"].quantile(0.25)
c = train_data["Crea"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["GOT"].quantile(0.75)
b = train_data["GOT"].quantile(0.25)
c = train_data["GOT"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["GPT"].quantile(0.75)
b = train_data["GPT"].quantile(0.25)
c = train_data["GPT"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["ALB"].quantile(0.75)
b = train_data["ALB"].quantile(0.25)
c = train_data["ALB"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["Na"].quantile(0.75)
b = train_data["Na"].quantile(0.25)
c = train_data["Na"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["K"].quantile(0.75)
b = train_data["K"].quantile(0.25)
c = train_data["K"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)

a = train_data["UA"].quantile(0.75)
b = train_data["UA"].quantile(0.25)
c = train_data["UA"]
c[(c >= (a-b)*1.5+a) | (c <= b-(a-b)*1.5)] = np.nan
c.fillna(np.nan, inplace=True)


#train_data.dropna(axis=1, how='any', inplace=True)

# min-max正規化


def min_max_scaler(x): return (x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x))


#train_data[['LOS']] = train_data[['LOS']].apply(min_max_scaler)
train_data[['LOS', 'OP_time_minute', 'CBC_WBC', 'CBC_RBC', 'CBC_HG', 'CBC_HT', 'CBC_MCV', 'CBC_MCH', 'CBC_MCHC', 'CBC_RDW', 'CBC_Platelet', 'CBC_RDWCV', 'BUN', 'Crea', 'GOT', 'GPT', 'ALB', 'Na', 'K', 'UA']] = train_data[[
    'LOS', 'OP_time_minute', 'CBC_WBC', 'CBC_RBC', 'CBC_HG', 'CBC_HT', 'CBC_MCV', 'CBC_MCH', 'CBC_MCHC', 'CBC_RDW', 'CBC_Platelet', 'CBC_RDWCV', 'BUN', 'Crea', 'GOT', 'GPT', 'ALB', 'Na', 'K', 'UA']].apply(min_max_scaler)


#train_data.fillna(value='?', inplace=True)
# train_data.corr(method ='pearson') '''pearson相關係數'''
# print(train_data)


# 寫入檔案
train_out0 = train_data[train_data.outcome == '0']
train_out1 = train_data[train_data.outcome == '1']

np.random.seed(10)
train_1000 = pd.concat([train_out0.sample(n=500), train_out1.sample(n=500)])
train_test = train_data.drop(train_1000.index)

t0 = train_test[train_test.outcome == '0']
t1 = train_test[train_test.outcome == '1']

test_100 = train_test
# print(train_1000)
train_1000.to_csv("test.data", encoding='utf-8',
                  index=False, header=False, na_rep='?')
test_100.to_csv("test.test", encoding='utf-8',
                index=False, header=False, na_rep='?')
