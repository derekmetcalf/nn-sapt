import os,sys

systems = ["NMA-Aniline-crystallographic"]

for system in systems:
    with open("%s_test_eval.csv"%system, "r") as read_file:
        with open("%s_trunc_eval.csv"%system, "w+") as write_file:
            next(read_file)
            next(read_file)
            next(read_file)
            next(read_file)
            next(read_file)
            next(read_file)
            next(read_file)
            for line in read_file:
                data = line.split(",")
                write_file.write("%s,%s,%s\n"%(data[0],data[1],data[2]))
