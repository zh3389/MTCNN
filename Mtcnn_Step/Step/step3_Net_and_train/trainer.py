from Mtcnn_Step.Step.step3_Net_and_train import Three_Net
from Mtcnn_Step.Step.step3_Net_and_train import data_torch
import torch.utils.data as data  # 处理数据集的包
import torchvision  # 数据集模块
import torch
import os  # 本文创建文件夹用


# the_model = torch.load(PATH)  # 加载模型

class Trainer():
    def __init__(self, net, size, batch, isCuda=False):
        '''
        :param net: 传入网络
        :param size: 传入网络训练的图大小,相当于选择训练集
        :param batch: 传入训练的批次大小
        '''
        self.net = net
        if isCuda:
            self.net.cuda()
        self.size = size
        self.batch = batch
        self.data = data_torch.FaceDataset(
            "D:/celeba/train_data/{}/".format(self.size))  # 将数据传入FaceDataset处理原始数据为 网络需要的数据
        self.loader = data.DataLoader(dataset=self.data, batch_size=batch, shuffle=True)  # shuffle(打乱数据)

    def facedata(self):
        '''
        :return: DataLoader
        '''
        return self.loader

    def forward(self, x):
        '''
        :param x: 训练的数据
        :return: 返回网络的输出值
        '''
        y = self.net.forward(x)
        return y

    def backward(self):
        '''
        :return: 返回当前网络的优化器
        '''
        return torch.optim.Adam(self.net.parameters())

    def total_loss(self, net_output, label):
        '''
        # net_output = face_classification, bounding_box
        # label = img_data, confidence, offset
        :param net_output:
        :param label:
        :return:
        '''
        face_out = net_output[0]  # #                       (预测) 取出预测置信度
        face_out = face_out.view(face_out.size(0), -1)  # # (预测) 置信度 降维
        box_out = net_output[1]  # #                        (预测) 取出预测 框
        box_out = box_out.view(face_out.size(0), -1)  # #   (预测) 框 降维
        MSE_Loss = torch.nn.MSELoss()  # #                  定义交叉熵损失函数
        BCE_Loss = torch.nn.BCELoss()
        CrossEntropyLoss = torch.nn.CrossEntropyLoss(size_average=False , ignore_index=-100, reduce=True)
        face_label = label[1]  # #                          (标签)取出标签 置信度 (0) or (1) or (2)
        box_label = label[2]  # #                           (标签) 取出标签 框  (x1, y1, x2, y2)
        face_index = torch.ne(label[1], 2)  # #             (标签) 筛选掉标签为2的样本 (类似bool类型的值)
        face_index = face_index.view(label[1].size(0))  # # (将index view成一维的)
        box_index = torch.ne(label[1], 0)  # #              (标签) 筛选掉标签为0的样本
        box_index = box_index.view(box_index.size(0))  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # 索引没问题, 原数据没问题
        a = face_label[face_index]  # 标签为 2 的置信度已筛选掉
        face_label = self.one_hot(a)  # 将标签转化为 one_hot形式  # 去除标签为 2 之后转换的onehot
        loss_1 = BCE_Loss(face_out[face_index], face_label)
        loss_2 = MSE_Loss(box_out[box_index], box_label[box_index])
        loss = loss_1 + loss_2
        return loss

    def load_save(self):
        pass

    def save_nn(self):
        '''
        :return: 保存当前网络的参数
        '''
        if not os.path.exists("save"):  # 如果没有该文件夹
            os.makedirs("save")  # 则创建
        return torch.save(self.net, "save/{}".format(self.size))  # 将网络的训练参数保存到本地

    def one_hot(self, data):
        class_num = 2
        label = torch.LongTensor(data.size(0), 1).random_() % class_num
        one_hot = torch.zeros(data.size(0), class_num).scatter_(1, label, 1)
        return one_hot


if __name__ == '__main__':
    batch = 32
    trainer_12 = Trainer(Three_Net.PNet(), 12, batch)  # 创建一个训练者对象
    trainer_24 = Trainer(Three_Net.RNet(), 24, batch)
    trainer_48 = Trainer(Three_Net.ONet(), 48, batch)
    # 标签输出二值, 使用交叉熵损失
    # 候选框输出四值, 使用欧几里德损失  三维空间(sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2))
    # 欧几里德损失 n维空间(sqrt((x1 - y1)**2 + (x2 - y2)**2 + ... + (xn - yn)**2))
    epoch = 0
    for _ in range(10000):
        for x12, x24, x48 in zip(trainer_12.facedata(), trainer_24.facedata(), trainer_48.facedata()):
            epoch += 1  # 计算训练次数
            # -----12-----12-----12-----12-----12

            y12_net_output = trainer_12.forward(x12[0])
            y12_opt = trainer_12.backward()
            y12_opt.zero_grad()
            loss_12 = trainer_12.total_loss(y12_net_output, x12)
            loss_12.backward()
            y12_opt.step()

            y24_net_output = trainer_24.forward(x24[0])
            y24_opt = trainer_24.backward()
            y24_opt.zero_grad()
            loss_24 = trainer_24.total_loss(y24_net_output, x24)
            loss_24.backward()
            y24_opt.step()

            y48_net_output = trainer_48.forward(x48[0])
            y48_opt = trainer_48.backward()
            y48_opt.zero_grad()
            loss_48 = trainer_48.total_loss(y48_net_output, x48)
            loss_48.backward()
            y48_opt.step()
            if epoch % 10 == 0:
                print("pnet:", loss_12)
                print("rnet:", loss_24)
                print("onet:", loss_48)
                print("-------------------------")


            # print(loss.item())

