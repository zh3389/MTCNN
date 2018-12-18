import numpy as np

a = np.arange(5*5).reshape(1, 5, 5)
print(a)

for i in a:
    print(i)
    print(i[0], i[1], i[2], i[3], i[4])

a = np.arange(9).reshape(3, 3)
b = np.arange(10, 19).reshape(3, 3)


c = np.stack((a, b), axis=0)
c = c.transpose((1, 0, 2))
c = c.reshape(-1, 6)
print(c)