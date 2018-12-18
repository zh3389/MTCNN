import numpy as np

a = np.arange(66, 75).reshape(3, 3)
b = np.array([100, 70, 50])
print(np.max(b))
print(a)
print(b)
print('=============')
y = np.maximum(a[:, 1], b[1])
print(y)