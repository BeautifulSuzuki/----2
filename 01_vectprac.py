# 数値解析2
import numpy as np

w = np.array([
    [-7, -2, 0],
    [1, 3, -2],
    [9, -2, 0],
    [-2, -3, 0],
    [-7, 0, 5]
])

x = np.array([1, 2, 3])
y = np.array([-5, -4, -3, -2, -1])

# Wx
print('w * x = ', w * x)
print('w @ x = ', w @ x)  # 行列積

# Wx + y
print('w @ x + y  = ', w @ x + y)
print('dot Wx + y = ', np.dot(w, x) + y)

# || Wx + y || _2
print('|| w @ x + y ||_2 = ', np.linalg.norm(w @ x + y))
print('|| w @ x + y ||_2 = ', np.linalg.norm(w @ x + y, ord=2))