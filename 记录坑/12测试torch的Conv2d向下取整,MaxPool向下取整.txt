import torch
import torch.nn as nn
import numpy as np

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main_nn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, 2, 1)
        )

    def forward(self, x):
        x = self.main_nn(x)
        return x


if __name__ == '__main__':
    net = Net()
    a = np.arange(48*48*3).reshape(1, 3, 48, 48)
    x = net.forward(torch.Tensor(a))
    print(x.shape)