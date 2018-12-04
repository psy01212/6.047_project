import gzip
##import tensorflow
import random
from os import listdir
from os.path import join

import time

start = time.time()

### there is no E060 or E064: only 127 types (not 129) ###
## total 1925196 lines in each file 

random.seed(42)
chrom = "chr21"
file_end = ".pval.signal.bedGraph.wig.gz"
datadir = join("..", "data", "all_data", "EXAMPLE", "CONVERTEDDATADIR")
mark_to_cell = {}

for file in listdir(datadir):
    if file.startswith(chrom):
        cell_type = file[len(chrom)+1:len(chrom)+5]
        mark = file[len(chrom)+6:-len(file_end)]
        if mark in mark_to_cell:
            mark_to_cell[mark].append(cell_type)
        else:
            mark_to_cell[mark] = [cell_type]

# x_mark is being used to predict y_mark
#   because H3K4me1 is there for all cell type, just use cell types that have H3K27ac
x_mark = "H3K27ac"
y_mark = "H3K4me1"

print(len(mark_to_cell[x_mark]))
print(len(mark_to_cell[y_mark]))

# shuffling all marks, first 60 are test, 18 are validation, 20 are test
all_types_shuffled = random.sample(mark_to_cell[x_mark], len(mark_to_cell[x_mark]))

all_test_xs = []
all_test_ys = []

for i in range(60):
    print(i)
    x_filename = chrom + "_" + all_types_shuffled[i] + "-" + x_mark + file_end
    with gzip.open(join(datadir, x_filename)) as f:
        # ignore first two lines (contains info about observed mark and chromosome, not actual signal)
        f.readline()
        f.readline()
        signals = []
        for line in f:
            signals.append(float(line.decode("utf-8")))
        all_test_xs.append(signals)

    y_filename = chrom + "_" + all_types_shuffled[i] + "-" + y_mark + file_end
    with gzip.open(join(datadir, y_filename)) as f:
        # ignore first two lines (contains info about observed mark and chromosome, not actual signal)
        f.readline()
        f.readline()
        signals = []
        for line in f:
            signals.append(float(line.decode("utf-8")))
        all_test_ys.append(signals)

all_validation_xs = []
all_validation_ys = []
for i in range(60, 78):
    print(i)
    x_filename = chrom + "_" + all_types_shuffled[i] + "-" + x_mark + file_end
    with gzip.open(join(datadir, x_filename)) as f:
        # ignore first two lines (contains info about observed mark and chromosome, not actual signal)
        f.readline()
        f.readline()
        signals = []
        for line in f:
            signals.append(float(line.decode("utf-8")))
        all_validation_xs.append(signals)

    y_filename = chrom + "_" + all_types_shuffled[i] + "-" + y_mark + file_end
    with gzip.open(join(datadir, y_filename)) as f:
        # ignore first two lines (contains info about observed mark and chromosome, not actual signal)
        f.readline()
        f.readline()
        signals = []
        for line in f:
            signals.append(float(line.decode("utf-8")))
        all_validation_ys.append(signals)

end = time.time()
print(end - start)

##with gzip.open("../data/all_data/EXAMPLE/CONVERTEDDATADIR/chr21_E003-DNase.pval.signal.bedGraph.wig.gz") as f:
##    numlines = 0
##    lastline = ""
##    f.readline()
##    f.readline()
##    x = []
##    for i in range(10):
##        line = f.readline().decode("utf-8")
##        x.append(float(line))
##    print(x)
    
