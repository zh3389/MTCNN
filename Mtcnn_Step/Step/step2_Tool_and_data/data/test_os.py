import os

path = "D:/123456"
if not os.path.exists(path):
    os.makedirs(path)

lir = [1, 2, 3, 4]
for _ in range(1):
    for i in range(1000):
        with open("D:/123456/1.txt", "a") as f:
            a = "{0}.jpg {1} {2} {3} {4} {5} \n".format(i, str(1), lir[0], lir[1], lir[2], lir[3])
            print(a)
            f.write(a)