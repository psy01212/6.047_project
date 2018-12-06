from os import listdir
from os.path import join

r2dir = "r2_scores"

pairs = []
max_r2s = []
median_r2s = []
for pair_file in listdir(r2dir):
    with open(join(r2dir, pair_file)) as f:
        # first line contains name of pair, second line is median r2, third line is max
        splitline = f.readline().split(": ")
        pairs.append(splitline[1][:-1])
        # for median/max, get rid of info about celltype by looking at substring
        splitline = f.readline().split(": ")
        median_r2s.append(float(splitline[1][7:-1]))
        splitline = f.readline().split(": ")
        max_r2s.append(float(splitline[1][7:-1]))

print pairs
print max_r2s
print median_r2s

print [x for _,x in sorted(zip(median_r2s,pairs), reverse=True)]
print [x for _,x in sorted(zip(max_r2s,pairs), reverse=True)]
