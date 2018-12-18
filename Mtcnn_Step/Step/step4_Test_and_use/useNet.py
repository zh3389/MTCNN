from Mtcnn_Step.Step.step4_Test_and_use import Three_Net
from Mtcnn_Step.Step.step4_Test_and_use import tool
import torch
import numpy as np
from PIL import Image


class use_Net:
    def __init__(self):
        '''
        初始化网络和权重
        '''
        self.pnet = Three_Net.PNet()
        self.pnet = torch.load(r"D:\python_prjs\Study\MTCNN\Step\step3_Net_and_train\save\12")
        self.pnet.eval()
        self.rnet = Three_Net.RNet()
        self.rnet = torch.load(r"D:\python_prjs\Study\MTCNN\Step\step3_Net_and_train\save\24")
        self.rnet.eval()
        self.onet = Three_Net.ONet()
        self.onet = torch.load(r"D:\python_prjs\Study\MTCNN\Step\step3_Net_and_train\save\48")
        self.rnet.eval()

    def _box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        '''
        :param start_index:
        :param offset:
        :param cls:
        :param scale:
        :param stride:
        :param side_len:
        :return:
        '''
        # print(start_index)  # tensor([ 42, 619])
        # print(offset.shape)  # torch.Size([4, 940, 1884])
        # print(cls.shape)  # torch.Size([2, 940, 1884])

        _x1 = (start_index[1] * stride) / scale  # 计算原图框的x1
        _y1 = (start_index[0] * stride) / scale  # 计算原图框的y1
        _x2 = (start_index[1] * stride + side_len) / scale  # 计算原图框的x2
        _y2 = (start_index[0] * stride + side_len) / scale  # 计算原图框的y2

        ow = _x2.detach().numpy() - _x1.detach().numpy()  # 计算原图框的 w
        oh = _y2.detach().numpy() - _y1.detach().numpy()  # 计算原图框的 h

        _offset = offset[:, start_index[0], start_index[1]].detach().numpy()  # 将图像的偏移量取出来[x1, y1, x2, y2]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls[1, start_index[0], start_index[1]]]

    def P_use(self, source_img):
        boxes = []  # 用来存放计算好的框
        img = source_img  # 将图片拷贝一份
        scale = 1  # 图片缩放比例

        while True:
            w, h = img.size
            print(w, h)
            if w < 12 or h < 12:
                break
            self.img = torch.Tensor(np.array(img, dtype=np.float) / 255. - 0.5)  # 打开一张图片 转为数组, 归一化
            self.img = self.img.permute((2, 0, 1)).unsqueeze(0)  # 将图片转置(hwc -> chw) 并升维 (加一个批次)
            self.face_out, self.box_out = self.pnet(
                self.img)  # torch.Size([1, 2, 940, 1884])  torch.Size([1, 4, 940, 1884])
            self.face_out = self.face_out.squeeze(0)  # torch.Size([2, 940, 1884])
            self.box_out = self.box_out.squeeze(0)  # torch.Size([4, 940, 1884])
            idxs = torch.nonzero(torch.gt(self.face_out[1, :, :], 0.6))  # torch.Size([344, 2]) x和y的位置(卷积后的图)
            for idx in idxs:
                boxes.append(self._box(idx, self.box_out, self.face_out, scale))
            print(np.array(boxes).shape)
            scale *= 0.7
            _w = int(w*scale)
            _h = int(h*scale)
            img = img.resize((_w, _h))
        return tool.nms(np.array(boxes), 0.5)

        # self.face_out = self.face_out.permute((0, 2, 3, 1)).squeeze(0)  # 将输出转置, 降维  torch.Size([940, 1884, 2])
        # self.box_out = self.box_out.permute((0, 2, 3, 1)).squeeze(0)  # 将输出转置, 降维  torch.Size([940, 1884, 4])
        # self.face_index = torch.lt(self.face_out[:, :, 0], 0.5)  # 取出置信度高的框的索引  torch.Size([485, 2])
        # self.face_out, self.box_out = (self.face_out[self.face_index])[:, 1], self.box_out[self.face_index]  # 将输出的值更新为需要的值

    def R_O_use(self):
        pass


if __name__ == '__main__':
    test_Net = use_Net()
    img = Image.open("2.jpg")
    test_Net.P_use(img)
