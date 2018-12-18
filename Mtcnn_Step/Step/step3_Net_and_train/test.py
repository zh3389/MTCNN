a = [[1, 1], [2, 2], [3, 3]]
b = [[4, 4], [5, 5], [6, 6]]
c = [[7, 7], [8, 8], [9, 9]]
d = zip(a, b, c)
# for i, (j,k), (l,m), (n,o) in enumerate(d):
#     print(j, k, l)

for i, j, k in d:
    print(i, j, k)
