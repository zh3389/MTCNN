import numpy as np

a = np.arange(2*4*5).reshape(1, 2, 4, 5)
print(a)
print(a[0, :, 0, 0])
a = a.transpose(0, 2, 3, 1)
print(a)