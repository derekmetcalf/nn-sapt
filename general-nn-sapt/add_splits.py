import os

files = os.listdir(".")
for xyz in files:
    with open(xyz, "r") as file:
        lines = file.readlines()
        if len(lines[1].split(",")) == 6:
            lines[1] = "%s,12\n"%(lines[1].replace("\n",""))
    with open("%s"%xyz, "w") as file:
        file.writelines(lines)
        print(file)
