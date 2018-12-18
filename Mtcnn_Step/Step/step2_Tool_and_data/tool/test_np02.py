import numpy as np


def nms(boxes):
    if boxes.shape[0] == 0:
        return np.array([])
    _boxes1 = np.argsort(-boxes[:, 0])
    _boxes2 = np.argsort(-boxes[:][0])
    _boxes3 = (-boxes[:, 4]).argsort()
    _boxes4 = (-boxes[:][4]).argsort()
    print(boxes[_boxes1])
    print(boxes[_boxes2])
    print(boxes[_boxes3])
    print(boxes[_boxes4])


# a1 = np.array([10, 10, 50, 50])  # Left up
# a2 = np.array([140, 10, 190, 60])  # Right up
#
# b1 = np.array([120, 30, 170, 80])  # Center
#
# c1 = np.array([10, 70, 60, 130])  # Left down
# c2 = np.array([150, 70, 200, 130])  # Right down
# boxes = np.array([a1, a2, b1, c1, c2])
boxes = np.arange(5 * 5).reshape(5, 5)
print(boxes)
nms(boxes)
# boxes2 = np.array([9, 5, 2, 15])
# nms(boxes2)
