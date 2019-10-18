#with open("don-and-acc-9-12_val.csv","r") as readfile:
#    with open("val_set.csv","w") as writefile:
#        lines = readfile.readlines()
#        for line in lines:
#            newline = line.split(",")[0] + "\n"
#            writefile.write(newline)

with open("don-and-acc-9-12_geoms.csv","r") as readfile:
    with open("val_set.csv","r") as valfile:
        with open("train_set.csv","w") as writefile:
            all_files = readfile.readlines()
            val_files = valfile.readlines()
            for filename in all_files:
                if filename not in val_files:
                    writefile.write(filename)
