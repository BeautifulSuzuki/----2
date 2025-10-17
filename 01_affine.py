# affine.py: Affine tranform
# 2318082 鈴木　祐亮
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

w1 = np.random.randn(5, 3)  # 5 x 3 matrix
b1 = np.random.randn(5)     # 5 dimensional vector
w2 = np.random.rand(4, 5)   # 4 x 5 matrix
b2 = np.random.randn(4)     # 4 dimensional
x  = np.random.randn(3)     # 3 dimensional vector

# h := W1 * x + b1
h = np.dot(w1, x) + b1
print('h = ', h)

# y := W2 * h + b2
y = np.dot(w2, h) + b2
print('y = ', y)

print('sigmoid(h) = ', sigmoid(h))

y = np.dot(w2, sigmoid(h)) + b2
print('y = ', y)
