with open("negative.txt", "r") as f1:
    cNames = f1.readlines()
    for i in range(0, len(cNames)):
        cNames[i] = "negative/" + cNames[i]

with open("negative_b.txt", "w") as f2:
    f2.writelines(cNames)
