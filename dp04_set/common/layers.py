# coomon/layers.py
import numpy as np
from common.functions import *
#from common.util import im2col, col2im

# P.33 MatMul class
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    
    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW # Deep copy
        return dx

# P.36 Sigmoid class
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
    
    def forward(self, x):
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx

# P.37 Affine class
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        #print('x.shape, W.shape, b.shape = ', x.shape, W.shape, b.shape)
        out = np.dot(x, W) + b
        self.x = x
        return out
    
    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis = 0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx

# P.38 Sotfmax with loss
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # output of Softmax
        self.t = None # Instructor data
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x) # in SciPy.special?
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # in case that instructor data is one-hot-vector
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx



