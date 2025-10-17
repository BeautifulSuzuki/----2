# two_layer_net.py: 2層NN
# P.43 -
import sys
sys.path.append('..')
import numpy as np 
# common/layers.py
from common.layers import Affine, Sigmoid, SoftmaxWithLoss

# TwoLayerNet class
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        
        # 重みとバイアスの初期化
        # 重みを小さいランダム値にすると学習が進みやすい
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # Layerの生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        # 全ての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    # predict関数
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x) # 予測を進める
        return x
    
    # forward関数
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss
    
    # backward関数
    def backward(self, dout = 1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
