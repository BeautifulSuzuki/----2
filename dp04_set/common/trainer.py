# trainer.py: 学習用 Trainer class
import sys
sys.path.append('..')
import numpy
import time
import matplotlib.pyplot as plt

# common/np.py
from common.np import *
from common.util import clip_grads

# 重みの重複を一つに集約し、重複分を勾配に加算する(p.117に解説あり)
def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]  # リストのコピー

    while True:
        find_flag = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flag = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                    params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flag = True
                    params.pop(j)
                    grads.pop(j)

                if find_flag: break
            if find_flag: break

        if not find_flag: break

    return params, grads

# Trainer class
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    # 順方向の学習（fitメソッド）
    def fit(self, x, t, max_epoch=10, batch_size=32, eval_interval=10):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        start_time = time.time()

        for epoch in range(max_epoch):
            # シャッフル
            idx = np.random.permutation(data_size)
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters * batch_size:(iters + 1) * batch_size]
                batch_t = t[iters * batch_size:(iters + 1) * batch_size]

                # 勾配算出
                loss = self.model.forward(batch_x, batch_t)
                self.model.backward()
                # grads = [p.grad for p in self.model.params]
                grads = self.model.grads
                params = self.model.params

                # 重複勾配の統合
                params, grads = remove_duplicate(params, grads)

                # 勾配クリッピング
                clip_grads(grads, 5.0)

                # パラメータ更新
                self.optimizer.update(params, grads)
                self.loss_list.append(loss)

                # 評価
                if eval_interval and iters % eval_interval == 0:
                    elapsed_time = time.time() - start_time
                    print(f"epoch:{epoch+1}, iter:{iters+1}/{max_iters}, "
                          f"loss:{loss:.4f}, time:{elapsed_time:.1f}s")

            self.current_epoch += 1

    # 学習曲線の描画
    def plot(self, png_filename='train_plot.png'):
        plt.plot(np.arange(len(self.loss_list)), self.loss_list)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('Training Curve')
        plt.grid(True)
        plt.savefig(png_filename)
        plt.show()