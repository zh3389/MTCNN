import torch
import numpy as np

a = np.arange(9).reshape(3, 3)
print(a)

b = torch.Tensor(a)
print(b)