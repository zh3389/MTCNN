
mobility_ratio = 0.2

for i in range(100):
    if i != 0:
        mobility_ratio *= 0.98 ** i
    print(mobility_ratio)

li = ["a", 1, 2]
print(li)
print(type(li[0]))