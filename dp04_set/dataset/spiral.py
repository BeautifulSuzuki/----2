# spiral.py: スパイラル点の分離
# https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/master/dataset/spiral.py
import numpy as np

# データの生成と読み込み
def load_data(CLS_NUM = 3, N = 100, DIM = 2, seed = 20201027):
    np.random.seed(seed) # 乱数の種をセット
#    N = 100 # クラスごとのサンプル数
#    DIM = 2 # データの次元数
#    CLS_NUM = 3 # クラス数
    print('CLS_NUM, N, DIM, seed = ', CLS_NUM, N, DIM, seed)

    x = np.zeros((N * CLS_NUM, DIM))
#   t = np.zeros((N * CLS_NUM, CLS_NUM), dtype = np.int)
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype = np.int64)

    for j in range(CLS_NUM):
        for i in range(N): # N * j, N * (j + 1)
            rate = i / N 
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            #theta = j * 4.0 + (CLS_NUM + 1) * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([
                radius * np.cos(theta),
                radius * np.sin(theta)
            ]).flatten()
            t[ix, j] = 1

    return x, t

# データの生成と読み込み(オリジナル)
def load_data_org(seed = 20201027):
    np.random.seed(seed) # 乱数の種をセット
    N = 100 # クラスごとのサンプル数
    DIM = 2 # データの次元数
    CLS_NUM = 3 # クラス数

    x = np.zeros((N * CLS_NUM, DIM))
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype = np.int)

    for j in range(CLS_NUM):
        for i in range(N): # N * j, N * (j + 1)
            rate = i / N 
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([
                radius * np.cos(theta),
                radius * np.sin(theta)
            ]).flatten()
            t[ix, j] = 1

    return x, t
