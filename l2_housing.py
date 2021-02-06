# 下載資料
# 要寫在 Python 檔案裡面
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok = True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.reqest.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()

# 用 pandas 載入資料
import pandas as pd

# def load_housing_data(housing_path = HOUSING_PATH):
#     csv_path = os.path.join(housing_path, "housing.csv")
#     return pd.read_csv(csv_path)
# housing = load_housing_data()
# print(housing.head())

# 載入資料
housing = pd.read_csv("housing.csv")

print("----------")
print("housing.head() 前五列資料：")
print(housing.head())
print("----------")
print("housing.info() 瞭解資料概況(包含總列數、非null值)：")
print(housing.info())
print("----------")
print("housing[\"ocean_proximity\"].value_counts() 瞭解它的值有哪些種類，以及多少地區屬於那些種類")
print(housing["ocean_proximity"].value_counts())
print("----------")
print("housing.describe() 顯示數值屬性的摘要")
print(housing.describe())
print("----------")

# 印出所有屬性的直方圖
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize = (20, 15))
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html
# plt.title("所有屬性的直方圖")##########################
plt.show()


# # test set - way 1:隨機排列取標籤分成測試組與對照組，但新增資料的變化取樣的彈性不足。
import numpy as np
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data)) # 隨機排列
#     test_set_size = int(len(data) * test_ratio) # 選取測試組個數
#     test_indices = shuffled_indices[: test_set_size] # 測試組
#     train_indices = shuffled_indices[test_set_size: ] # 訓練組
#     return data.iloc[train_indices], data.iloc[test_indices] 
# # call function
# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set))
# print(len(test_set))

# test set - way 2:
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42) # 20% 的測試組
print("訓練組樣本數：", len(train_set))
print("測試組樣本數：", len(test_set))

# 分層抽樣
housing["income_cat"] = pd.cut(housing["median_income"], bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels = [1, 2, 3, 4, 5])
housing["income_cat"].hist()
plt.show()

# StratifiedShuffleSplit 分層隨機拆分
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    # 檢查分類比率
    print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))


# 移除 income_cat 屬性，讓資料回到原始狀態
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True)

# 先建立副本
housing = strat_train_set.copy()

# 將地理資料視覺化
housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha = 0.1)
plt.show()