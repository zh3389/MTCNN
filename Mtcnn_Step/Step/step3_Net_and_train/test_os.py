import os

path = r"D:/test/"
name = "utils.py"
name2 = "utils2.py"

c = []
c.extend(open(os.path.join(path, name)).readlines())
c.extend(open(os.path.join(path, name)).readlines())
print(c)