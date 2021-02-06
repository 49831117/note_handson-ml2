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
plt.show()
