import os

files = os.listdir(".")
for xyz in files:
    with open(xyz, "r") as file:
        lines = file.readlines()
        if len(lines[1].split(",")) == 6:
            lines[1] = "%s,12"%lines[1]
    with open("spike_%s"%xyz, "w") as file:
        file.writelines(lines)
        print(file)
