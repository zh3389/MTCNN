from Mtcnn_Step.Step.step3_Net_and_train import Three_Net
from Mtcnn_Step.Step.step3_Net_and_train import data_torch
import torch.utils.data as data  # 处理数据集的包
import torchvision  # 数据集模块
import torch
import os  # 本文创建文件夹用
import numpy as np
import threading

# the_model = torch.load(PATH)  # 加载模型

class Trainer():
    def __init__(self, net, img_size, lr=1e-3, isCuda=False):
        '''
        :param net: 传入网络
        :param size: 传入网络训练的图大小,相当于选择训练集
        :param batch: 传入训练的批次大小
        '''
        self.net = net
        self.img_size = img_size
        self.isCuda = isCuda
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        if self.isCuda:
            self.net.cuda()

    def train(self, batch, isLoad=False, isSave=False):
        '''
        :param batch: 训练的批次
        :return: 训练网络
        '''
        self.data = data_torch.FaceDataset(
            "D:/celeba/train_data/{}/".format(self.img_size))  # 将数据传入FaceDataset处理原始数据为 网络需要的数据
        self.loader = data.DataLoader(dataset=self.data, batch_size=batch, shuffle=True)  # shuffle(打乱数据)
        MSE_Loss = torch.nn.MSELoss()
        BCE_Loss = torch.nn.BCELoss()

        if isLoad:
            self.net = torch.load("save/{}".format(self.img_size))

        while True:
            for i, (self.img_data, self.face_label, self.box_label) in enumerate(self.loader):

                if self.isCuda:
                    self.img_data.cuda()  # shape([batch, 3, 12, 12])  # <class 'torch.Tensor'>
                    self.face_label.cuda()  # shape([batch, 1])  # <class 'torch.Tensor'>
                    self.box_label.cuda()  # shape([batch, 4])  # <class 'torch.Tensor'>
                self.face_out, self.box_out = self.net(
                    self.img_data)  # shape([batch, 2, 1, 1]), shape([batch, 4, 1, 1])  # <class 'torch.Tensor'>
                self.face_out = self.face_out.view(self.face_out.size(0), -1)
                self.box_out = self.box_out.view(self.box_out.size(0), -1)
                # shape([batch])  # <class 'torch.Tensor'>  # 筛选掉 置信度为 2 的人脸(中等样本)
                self.face_index = torch.ne(self.face_label, 2).view(self.face_label.size(0))
                # shape([batch])  # <class 'torch.Tensor'>  # 筛选掉 置信度为 0 的人脸(负样本)
                self.box_index = torch.ne(self.face_label, 0).view(self.face_label.size(0))
                self.face_label = self.one_hot(self.face_label[self.face_index])  # 将标签转换为one_hot形式
                self.loss_1 = BCE_Loss(self.face_out[self.face_index], self.face_label)
                self.loss_2 = MSE_Loss(self.box_out[self.box_index], self.box_label[self.box_index])
                self.loss = self.loss_1 + self.loss_2

                self.opt.zero_grad()
                self.loss.backward()
                self.opt.step()

                if i % 100 == 0:
                    print(self.loss.item())
                    if isSave:
                        if not os.path.exists("save"):  # 如果没有该文件夹
                            os.makedirs("save")  # 则创建
                        torch.save(self.net, "save/{}".format(self.img_size))  # 将网络的训练参数保存到本地

    def one_hot(self, data):
        '''
        :param data: 传入需要转为one-hot的数据
        :return: 转换好的one-hot
        '''
        data = np.array(data)
        data = (np.arange(2) == data[:, None]).astype(np.float)
        data = torch.Tensor(data)
        data = data.view(data.size(0), -1)
        return data


if __name__ == '__main__':
    # batch = 512
    # trainer_12 = Trainer(Three_Net.PNet(), 12)  # 创建一个训练者对象
    # trainer_24 = Trainer(Three_Net.RNet(), 24)
    # trainer_48 = Trainer(Three_Net.ONet(), 48)
    # 标签输出二值, 使用交叉熵损失
    # 候选框输出四值, 使用欧几里德损失  三维空间(sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2))
    # 欧几里德损失 n维空间(sqrt((x1 - y1)**2 + (x2 - y2)**2 + ... + (xn - yn)**2))
    # trainer_12.train(batch, isLoad=False, isSave=False)
    # trainer_24.train(batch, isLoad=True, isSave=True)
    # trainer_48.train(batch, isLoad=True, isSave=True)

    batch = 512
    trainer_12 = Trainer(Three_Net.PNet(), 12)  # 创建一个训练者对象
    trainer_24 = Trainer(Three_Net.RNet(), 24)
    trainer_48 = Trainer(Three_Net.ONet(), 48)
    thread1 = threading.Thread(target=trainer_12.train, args=(batch, ), name='P网络')
    thread2 = threading.Thread(target=trainer_24.train, args=(batch, ), name='R网络')  # 再创建一个线程 设置线程名为 2
    thread3 = threading.Thread(target=trainer_48.train, args=(batch, ), name='O网络')  # 创建一个名为 3 的线程
    thread1.start()  # 启动线程 1
    # thread2.start()  # 启动线程 2
    # thread3.start()  # 启动线程 3