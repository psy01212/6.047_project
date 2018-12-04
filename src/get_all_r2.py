from os import listdir
from os.path import join
import gzip
import statistics
from sklearn import linear_model
from sklearn.metrics import r2_score

chrom = "chr21"
file_end = ".pval.signal.bedGraph.wig.gz"
datadir = join("..", "data", "all_data", "EXAMPLE", "CONVERTEDDATADIR")
savedir = join("..", "data", "r2_scores")
mark_to_cell = {}

for file in listdir(datadir):
    if file.startswith(chrom):
        cell_type = file[len(chrom)+1:len(chrom)+5]
        mark = file[len(chrom)+6:-len(file_end)]
        if mark in mark_to_cell:
            mark_to_cell[mark].append(cell_type)
        else:
            mark_to_cell[mark] = [cell_type]

### all_marks order: ['H3K9me3', 'DNase', 'H3K27ac', 'H3K4me3', 'H3K36me3', 'H3K9ac', 'H3K4me1', 'H3K27me3']
all_marks = []
for mark in mark_to_cell:
    all_marks.append(mark)
print(all_marks)

for mark_i in range(len(all_marks)):
    for mark_j in range(mark_i+1, len(all_marks)):
        pair = all_marks[mark_i] + "_" + all_marks[mark_j]
        pair_file = pair + ".txt"
        print(pair_file)
        # if this pair of marks has already been analyzed, skip
        if pair_file in listdir(savedir):
            pass
        else:
            all_cells = []
            all_r2s = []
            for cell in mark_to_cell[all_marks[mark_i]]:
                if cell in mark_to_cell[all_marks[mark_j]]:
                    print(cell)
                    x_filename = chrom + "_" + cell + "-" + all_marks[mark_i] + file_end
                    y_filename = chrom + "_" + cell + "-" + all_marks[mark_j] + file_end
                    all_xs = []
                    all_ys= []
                    with gzip.open(join(datadir, x_filename)) as f:
                        f.readline()
                        f.readline()
                        for line in f:
                            all_xs.append([float(line.decode("utf-8"))])
                    with gzip.open(join(datadir, y_filename)) as f:
                        f.readline()
                        f.readline()
                        for line in f:
                            all_ys.append([float(line.decode("utf-8"))])
                    regr = linear_model.LinearRegression()
                    regr.fit(all_xs, all_ys)
                    predicted_ys = regr.predict(all_xs)
                    r2_value = r2_score(all_ys, predicted_ys)
                    all_cells.append(cell)
                    all_r2s.append(r2_value)
            ## after looking at all celltypes that has values for the pair of marks, find median, max, and min
            median_r2 = statistics.median_high(all_r2s)
            median_cell = all_cells[all_r2s.index(median_r2)]
            min_r2 = min(all_r2s)
            min_cell = all_cells[all_r2s.index(min_r2)]
            max_r2 = max(all_r2s)
            max_cell = all_cells[all_r2s.index(max_r2)]
            with open(join(savedir, pair_file), 'w') as f:
                f.write("Pair: " + pair + "\n")
                f.write("Median R2: " + median_cell + " - " + str(median_r2) + "\n")
                f.write("Max R2: " + max_cell + " - " + str(max_r2) + "\n")
                f.write("Min R2: " + min_cell + " - " + str(min_r2) + "\n")
                f.write("All R2 values:\n")
                for i in range(len(all_cells)):
                    f.write(all_cells[i] + " - " + str(all_r2s[i]) + "\n")
                    
