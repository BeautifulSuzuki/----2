# train_custom_loop.py: 学習用コード
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt

# dataset/spiral.py
from dataset import spiral
# common/optimizer.py
from common.optimizer import SGD
# two_layer_net.py
from two_layer_net import TwoLayerNet


# (1) Hyper parametersの設定
max_epoch = 300 # epochは学習の単位:「1 epoch = 全てのデータを1回学習」
batch_size = 30
hidden_size = 10
learning_rate = 1.0


# (2) データの読み込み、モデルとオプティマイザの設定
x, t = spiral.load_data()
model = TwoLayerNet(input_size = 2, hidden_size = hidden_size, output_size = 3)
optimizer = SGD(lr = learning_rate)
