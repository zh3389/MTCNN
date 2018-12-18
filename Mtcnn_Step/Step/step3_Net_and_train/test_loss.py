import torch
import torch.nn as nn
import numpy as np

# BCE_Loss = nn.BCELoss()  # 输入应该在 0~1之间
# Softmax_Loss = nn.Softmax()  # TypeError: forward() takes 2 positional arguments but 3 were given
# MSE_Loss = nn.MSELoss()  # (100.)
CrossEntropy_Loss = nn.CrossEntropyLoss()  # RuntimeError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
Softplus_Loss = nn.Softplus()  # TypeError: forward() takes 2 positional arguments but 3 were given

# a = torch.Tensor(np.random.random(3*3).reshape(3*3))
# print(a)
# b = torch.Tensor(np.arange(10, 10 + 3*3).reshape(3*3))
# print(b)
# c = CrossEntropy_Loss(a, b)
# print(c)

# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# print(input)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(target)
# output = loss(input, target)
# print(output)
# output.backward()

target = torch.empty(1, dtype=torch.long).random_(4)
print(target)