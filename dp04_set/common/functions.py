# functions.py
import numpy as np 

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximun(0, x)

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis = 1, keepdims = True)
        x = np.exp(x)
        x /= x.sum(axis = 1, keepdims = True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    
    return x

# def cross_entropy_error(y, t)
def cross_entropy_error(y, t):
	#print('b y.ndim, size, shape, batch_size = ', y.ndim, y.size, y.shape)
	#print('b t.ndim, size, shape, batch_size = ', t.ndim, t.size, t.shape)
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	# 教師データがone-hot-vectorの場合，正解ラベルのインデックスに変換
	if t.size == y.size:
		t = t.argmax(axis = 1)

	batch_size = y.shape[0]
	#print('a y.ndim, size, shape, batch_size = ', y.ndim, y.size, y.shape, batch_size)
	#print('a t.ndim, size, shape, batch_size = ', t.ndim, t.size, t.shape, batch_size)

	return -np.sum(np.log(y[np.arange(batch_size), t] + 1.0e-7)) / batch_size
