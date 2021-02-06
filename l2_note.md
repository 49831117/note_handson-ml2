# Scikit - Learn, Keras, Tensorflow
----

## 第二章 端對端機器學習專案

### 步驟
1. 了解大局
2. 取得資料
3. 探索資料並視覺化，以取得見解
4. 準備資料，供機器學習使用
5. 選擇模型並訓練它
6. 調整模型
7. 展示解決方案
8. 啟動、監視與維護系統

#### 真實的資料
- 熱門的開放資料存放區
1. [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/)
2. [Kaggle datasets](https://www.kaggle.com/datasets)
3. [Amazon's AWS datasets](https://registry.opendata.aws/)
- 中繼網站 - 列出許多開放資料存放區
1. [Data Portals](http://dataportals.org)
2. [OpenDataMonitor](http://opendatamonitor.eu/)
3. [Quandl](http://quandl.com/)
- 羅列熱門開放資料存放區
1. [維基百科 - 機器學習資料組清單](https://homl.info/9)
2. [Quora.com](https://homl.info/10)
3. [reddit 的資料組](https://www.reddit.com/r/datasets)

#### 制定問題
- 監督（有標籤）、無監督、強化學習
- 分類任務、回歸任務、其他
- 批次學習（資料不會持續流入系統）、線上學習技術

#### 選擇性能指標
> 以下兩種皆為衡量兩向量之間距離的方式（預測＆目標的距離）
1. 均方根誤差 RMSE
2. 平均絕對誤差 MAE 

> 可以使用各種距離指標，或範數（norm）。
- 歐幾里得範數，計算平方和的根（RMSE）：*l2 norm* 
- 曼哈頓範數，絕對值的和（MAE）：*l1 norm*

#### 檢查假設

#### 取得資料
- 常用模組：Numpy、pandas、Matplotlib、Scikit-Learn
- 下載資料

```python
# 要寫在 Python 檔案裡面
import os
import tarfile
import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = download_root + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url = HOUSING_URL, housing_path = HOUSING_PATH):
    os.makedirs(housing_path, exist_ok = True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.reqest.urlretrieve(housing_url, tgz_path)
    housing_tqz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()
```
    
接著用 pandas 載入資料

```python
import pandas

def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
```
  1. `housing.head()` 查看前五列資料。
  2. `housing.info()` 瞭解資料概況。
    - 總列數、每個屬性的型態、非 null 值的數量。
  3. `housing["ocean_proximity"].value_counts()` 瞭解它的值有哪些種類，以及多少地區屬於那些種類。
  4. 感受資料的型態
     1. `housing.describe()` 顯示數值屬性的摘要，忽略了 null 的值。
     2. 畫出直方圖
        可以每次畫出一個屬性，也可以對整個資料組呼叫 `hist()` 幫每個資料數值屬性畫出直方圖。
        ```python
        import matplotlib.pyplot as plot
        housing.hist(bins = 50, figsize = (20, 15))
        plt.show()
        ```

#### 建立測試組
**避免 overfit**
1. 「資料窺探（data snooping）偏差」
2. 測試組：
    1. 隨機選擇實例（20% 或更少）
        ```python
        import numpy as np

        def split_train_test(data, test_ratio):
            shuffled_indices = np.random.permutation(len(data)) # 隨機排列
            test_set_size = int(len(data) * test_ratio) # 選取測試組個數
            test_indices = shuffled_indices[: test_set_size] # 測試組
            train_indices = shuffled_indices[test_set_size: ] # 訓練組
            return data.iloc[train_indices], data.iloc[test_indices] 
        # call function
        train_set, test_set = split_train_test(housing, 0.2)
        len(train_set)
        len(test_set)
        ```
    2. 缺點：無法完美隔開測試組，應該避免看到整組資料。
        
        **解決方式：**
        way1. 
        第一次執行時儲存測試組，後續執行再載入上述方式。

        way2. 
        設定亂數產生的種子
            eg. `np.random.seed()`
                再呼叫 `np.random.permutation()`讓它永遠產生洗亂過的索引。
        - 不過上述兩種方式將於下次取得更新過的資料組時失效。
        - 為了修正此問題，常見的作法是使用各個實例的辨識碼的雜湊，放入測試組。新的測試組將包含 20% 新實例，但不包含之前已經在訓練組裡面的任何實例。
        ```python
        from zlib import crc32

        def test_set_check(identifier, test_ratio):
            return crc32(np.int64(identifier)) & 0xffffffff <test_ratio * 2**32
        def split_train_test_by_id(data, test_ratio, id_column):
            ids = data[id_column]
            in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
            return data.loc[~in_test_set], data.loc[in_test_set]
        ```
        若資料若無識別碼欄位，可將索引當成 ID 來使用。
        ```python
        housing_with_id = housing.reset_index() # 加入 index 欄位
        train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
        ```
        但要將新資料附加到資料組的最後面，且不刪除任何資料，確保使用不改變的特徵建立識別碼。
        ```python
        housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
        train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
        ```
        
        - **Scikit-Learn `train_test_split()`** 和 `split_train_test()` 很像，也有額外功能。
          1. 有 `random_state` 參數，設定亂數產生器的種子。
          2. 將多個列數相同的資料組傳給它，它會在同一個索引拆開它們。
            ```python
            from sklearn.model_selection import train_test_split
            train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
            ```
    3. 抽樣需具有代表性。
        eg. 分層抽樣 `pandas.cut()`，但要注意層數不宜過多、每一層數量夠多。
        ```python
        housing["incom_cat"] = pd.cut(housing["Median_income"], bins = [0., 1.5, 3., 4.5, 6., np.inf], labels = [1, 2, 3, 4, 5])
        print(housing["income_cat"].hist())
        ```
    4. 透過 `Scikit-Learn` 的 `StratifiedShuffleSplit`
        - [sklearn.model_selection.StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)
        ```python
        from sklearn.model_selection import StratifiedShuffleSplit

        split = StratifiedShuffleSplit(n_split = 1, test_size = 0.2, random_state = 42)
        
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]
        
        # 檢查分類比率
        print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
        ```
    5. 移除 `income_cat` 屬性，讓資料回到原始狀態。
        ```python
        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis = 1, inplace = True)
        ```
3. 建立測試組的複本，以免破壞訓練組。
    ```python
    housing = strat_train_set.copy()
    ```
4. 將地理資料視覺化
   ```python
   housing.plot(kind = "scatter", x = "longitide", y = "latitude", alpha = 0.1)
   ```