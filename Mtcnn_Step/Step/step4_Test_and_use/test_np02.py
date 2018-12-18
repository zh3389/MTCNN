import numpy as np

a = np.array([])
for i in range(10):
    # print(np.array([i]))
    np.stack((a, np.array([i])), axis=0)
print(a)