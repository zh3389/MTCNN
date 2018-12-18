import torch.nn as nn

class PNet(nn.Module):
    '''
    :projective input: 12*12*3的图片(NCHW)
    :param 1: 12*12*3
    :param 2: 5*5*10
    :param 3: 3*3*16
    :param 4: 1*1*32
    :return output: 1*1*2的置信度, 1*1*4的目标框
    '''

    def __init__(self):
        super(PNet, self).__init__()
        self.main_nn = nn.Sequential(
            # (batch, 3, 12, 12) -> (batch, 10, 10, 10)
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # (batch, 10, 10, 10) -> (batch, 10, 5, 5)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # (batch, 10, 5, 5) -> (batch, 16, 3, 3)
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # (batch, 16, 3, 3) -> (batch, 32, 1, 1)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.PReLU()
        )
        # (batch, 32, 1, 1) -> (batch, 2, 1, 1)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.Softmax2d()
        )
        # (batch, 32, 1, 1) -> (batch, 4, 1, 1)
        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        '''
        :param x: 传入数据将 数据传入网络计算后返回
        :return: 网络最后一层的输出值
        '''
        x = self.main_nn(x)
        face_classification = self.conv_2(x)
        bounding_box = self.conv_4(x)
        return face_classification, bounding_box


class RNet(nn.Module):
    '''
    :projective input: 24*24*3(NCHW)
    :param 1: 24*24*3
    :param 2: 11*11*28
    :param 3: 4*4*48
    :param 4: 3*3*64
    :param 5: 128
    :return output: 1*2的置信度, 1*4的目标框
    '''

    def __init__(self):
        super(RNet, self).__init__()
        self.main_nn = nn.Sequential(
            # (batch, 3, 24, 24) -> (batch, 28, 22, 22)
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # (batch, 28, 22, 22) -> (batch, 28, 11, 11)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 向下取整, 会报维度错误
            # (batch, 28, 11, 11) -> (batch, 48, 9, 9)
            nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # (batch, 48, 9, 9) -> (batch, 48, 4, 4)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # (batch, 48, 4, 4) -> (batch, 64, 3, 3)
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2, stride=1, padding=0),
            nn.PReLU()

        )
        # (batch, 3*3*64) -> (batch, 128)
        self.FC_128 = nn.Sequential(
            nn.Linear(in_features=3 * 3 * 64, out_features=128),
            nn.PReLU()
        )
        # (batch, 128) -> (batch, 2)
        self.FC_2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=2),
            nn.Softmax(dim=1)
        )
        # (batch, 128) -> (batch, 4)
        self.FC_4 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        '''
        :param x: 传入数据将 数据传入网络计算后返回
        :return: 网络最后一层的输出值
        '''
        x = self.main_nn(x)
        x = x.view(x.size(0), -1)
        x = self.FC_128(x)
        face_classification = self.FC_2(x)
        bounding_box = self.FC_4(x)
        return face_classification, bounding_box


class ONet(nn.Module):
    '''
    :projective input: 48*48*3(NCHW)
    :param 1: 48*48*3
    :param 2: 23*23*32
    :param 3: 10*10*64
    :param 4: 4*4*64
    :param 5: 3*3*128
    :param 6: 256
    :return output: 1*2的置信度, 1*4的目标框
    '''

    def __init__(self):
        super(ONet, self).__init__()
        self.main_nn = nn.Sequential(
            # (batch, 3, 48, 48) -> (batch, 32, 46, 46)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # (batch, 32, 46, 46) -> (batch, 32, 23, 23) (padding=1)(2, 3报错)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # (batch, 32, 23, 23) -> (batch, 64, 21, 21)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # (batch, 64, 21, 21) -> (batch, 64, 10, 10)  (padding=0) (1时= batch, 64, 11, 11) (2报错)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            # (batch, 64, 10, 10) -> (batch, 64, 8, 8)
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            # (batch, 64, 8, 8) -> (batch, 64, 4, 4)
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # (batch, 64, 4, 4) -> (batch, 128, 3, 3)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=0),
            nn.PReLU(),
        )
        # (batch, 3*3*128) -> (batch, 256)
        self.FC_256 = nn.Sequential(
            nn.Linear(in_features=3 * 3 * 128, out_features=256),
            nn.PReLU()
        )
        # (batch, 256) -> (batch, 2)
        self.FC_2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=2),
            nn.Softmax(dim=1)
        )
        # (batch, 256) -> (batch, 4)
        self.FC_4 = nn.Linear(in_features=256, out_features=4)

    def forward(self, x):
        '''
        :param x: 传入数据将 数据传入网络计算后返回
        :return: 网络最后一层的输出值
        '''
        x = self.main_nn(x)
        x = x.view(x.size(0), -1)
        x = self.FC_256(x)
        face_classification = self.FC_2(x)
        bounding_box = self.FC_4(x)
        return face_classification, bounding_box


if __name__ == '__main__':
    '''
    执行此文件查看 网络是否搭建有问题
    '''
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
