# spiral_dataset_show.py: spiral.py生成のデータを表示する
import sys  # システム設定用
sys.path.append('..')  # 親ディレクトリにパスを通す
from dataset import spiral  # dataset/spiral.pyを読み込む
import matplotlib.pyplot as plt  # グラフ描画用

# spiralデータを読み込み、xとtのデータ形式を確認
x, t = spiral.load_data()
print('x -> ', x.shape, x.dtype)  # (300, 2) float64... 2次元300個のdouble型データ
print('t -> ', t.shape, t.dtype)  # (300, 3) int64... 3次元300個のlong型データ
# print('x = \n', x[:,0], x[:,1])

# グラフ表示
fig, ax = plt.subplots()

# 一色
ax.scatter(x[:, 0], x[:, 1])  # 散布図

# クラスごと塗分け
CLS_NUM, N = 3, 100
markers = ['o', 'x', '^', '+', '*']  # 5クラスまで対応
for i in range(CLS_NUM):  # CLS_NUM
    ax.scatter(x[i * N: (i + 1) * N, 0], x[i * N: (i + 1) * N, 1], s = 40, marker = markers[i])

fig.suptitle('Spiral data: x')
plt.show()  # グラフの画面表示
fig.savefig('spiral' + str(CLS_NUM) + '_' + str(N) + '.png')
