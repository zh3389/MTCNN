with open("test.txt") as f:
    for i, j in enumerate(f.readlines()):
        if i < 2:
            continue
        print(j)  # j 输出正常, 输出每行数据

        a, b, c, d, e = j.split()  # 数据拆分正常 将数据拆分装进列表 并去除了空格和换行符
        print(a, b, c, d, e)

        # 练习一下其它方法
