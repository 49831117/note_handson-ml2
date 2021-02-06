# 取得資料組
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version = 1)
print("mnist.keys()", mnist.keys())

# 觀察資料
X, y = mnist["data"], mnist["target"]
print("X.shape", X.shape)
# 可得 (70000, 784)，70000 張圖像、784 個特徵
print("y.shape", y.shape)
# 可得 (70000,)

# 抓取一個實例的特徵向量，將形狀改成 28*28 陣列，並印出觀察
import matplotlib as mpl
import matplotlib.pyplot as plt
################################### 跑不動
# some_digit = X[0]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap = "binary")
# plt.axis("off")
# plt.show()

