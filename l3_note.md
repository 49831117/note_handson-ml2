# Scikit - Learn, Keras, Tensorflow
----

## 第三章 分類

### MNIST
1. 取得資料組
    ```python  
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version = 1)
    mnist.key()
    ```
2. `DESCR` 描述資料組，`X` 為 data，`y` 為 target
   - 共有70000張圖片，