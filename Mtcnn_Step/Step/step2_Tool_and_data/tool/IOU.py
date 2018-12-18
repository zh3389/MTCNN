import numpy as np


def iou(box, boxes, isBase=True):
    '''
    :objective:     计算 box 和 boxes (批量)框的交并比.
    :param box:     传入矩形框的数据为: 左上角点 和 右下角点 (x1, y1, x2, y2, *args).
    :param boxes:   传入批量矩形框的数据为: [[x1, y1, x2, y2, *args], [x1, y1, x2, y2, *args]...].
    :param isBase:  True: 使用普通的交并比. False: 使用最小交并比
    :return:        返回一个数组, box 和 boxes里每个框的IOU(交并比).
    '''
    box_area = (box[2] - box[0]) * (box[3] - box[1])  # box 框的面积
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # 批量矩形框的面积
    # 小小算法部分 (计算出 单个矩形框 和 批量矩形框 的交集框(左上角点和右下角点))
    xx1 = np.maximum(box[0], boxes[:, 0])  # (求最大)box 的 x1 和 boxes 每行的 x1 做比较(广播比较)
    yy1 = np.maximum(box[1], boxes[:, 1])  # (求最大)box y1 -> boxes y1
    xx2 = np.minimum(box[2], boxes[:, 2])  # (求最小)box x2 -> boxes x2
    yy2 = np.minimum(box[3], boxes[:, 3])  # (求最小)box y2 -> boxes y2
    # print(xx1, yy1, xx2, yy2)  # 调试用, 观察数据是否正常
    # 现在已知 交集框 的左上角点和右下角点 计算交集框的面积(编写时注意: 有无交集的情况)
    # 如果 右下角点减去左上角点为负说明没有交集部分 则取0 (否则 则为矩形的长和宽) 将长和宽相乘为交集矩形的面积
    intersection_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)  # 将这个批量交集框的 数组进行求面积

    # 编写求 交集与并集之比, 交集与最小矩形面积之比
    if isBase:  # 普通交并比 (交集 / 并集(并集 = 两个矩形面积和 - 交集))
        return intersection_area / (box_area + boxes_area - intersection_area)  # 返回的是一个列表,计算出 矩形框 和 批量矩形框的交并比
    else:  # 交集与最小矩形面积之比 (交集 / 最小矩形面积)
        return intersection_area / np.minimum(box_area, boxes_area)  # 返回的是一个列表, 计算出 矩形框 和 批量矩形框的 最小交并比


def test_iou():
    '''
    :objective: unit testing
    :param :    None
    :return:    None
    '''
    a1 = np.array([10, 10, 50, 50])  # Left up
    a2 = np.array([140, 10, 190, 60])  # Right up

    b1 = np.array([120, 30, 170, 80])  # Center

    c1 = np.array([10, 70, 60, 130])  # Left down
    c2 = np.array([150, 70, 200, 130])  # Right down
    print(iou(np.array(b1), np.array([a1, a2, c1, c2])))
    print(iou(np.array(b1), np.array([a1, a2, c1, c2]), isBase=False))


if __name__ == '__main__':
    test_iou()  # test
