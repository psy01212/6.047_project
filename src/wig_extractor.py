# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:58:53 2018

@author: luisxavierramostormo
"""

import numpy as np
import gzip
import time
import os

# tells you which samples have which chromatin modification data
DATA_INFO_DIR = "../data/all_data/EXAMPLE/tier1_samplemarktable.txt"

# data of celltype C with modification H is stored in 
DATA_EXTENSION = ".pval.signal.bedGraph.wig.gz"

DATA_DIR_W_CHROM = "../data/all_data/EXAMPLE/CONVERTEDDATADIR/chr21_" # just add celltype, marker, and extension

DATA_DIR = "../data/all_data/EXAMPLE/CONVERTEDDATADIR/" # folder with all data. Doesn't specify chromosome

DATA_POINT_LEN = 1925196

DATA_POINTS = 127



def extract_cell_type_marks_avail():
    """
    returns mapping of {cell_type_num : {set of modifications available}}
    WORKING
    """
    out = {}
    with open(DATA_INFO_DIR) as f:
        for line in f:
            line = line.replace("\t", " ")
            cell_type, modification, filename = line.split(" ")
            filename = filename.strip()
            if cell_type not in out:
                out[cell_type] = set()
            out[cell_type].add(modification)
        f.close()
    return out

def extract_wig_gz(filename, split = None):
    """
    WORKING
    filename must be .gz
    Outputs np 2D array with data all in one row
    """
    out = []
    i = 2 # trim first two rows that contain all the same info
    with gzip.open(filename) as f:
        for line in f:
            if i > 0:
                i -= 1
                continue
            out.append(float(line.strip())) # trim whitespace. Add datapoints as floats
        f.close()
    out = np.array(out)
    
    if split == None:
        return np.array([out])
    
    # split into sections if argument specified
    sections = np.array_split(out, split)
    out = []
    for section in sections:
        out.append(max(section))
    return np.array([out])
    
#print(extract_wig_gz("../data/example_data/chr21_E001-H3K4me1.pval.signal.bedGraph.wig.gz", split = 100))
    
    

def load_training_data(mod1, mod2, split = 1000):
    """
    Extract data from many files and put it all together. Split it in "bites" that
    fit as input of CNN. Should be numpy array of 2D arrays, each of which is an input
    
    inputs are 1xN, all of the same marker
    
    inputs for cnn are markers of mod1
    outputs are markers of mod2
    
    function returns: (inputs, outputs), for all cell_types that have both markers
    """
    inputs = []
    outputs = []
    cell_types = []
    cell_type_mods = extract_cell_type_marks_avail()
    for cell_type in cell_type_mods:
        if mod1 in cell_type_mods[cell_type] and mod2 in cell_type_mods[cell_type]:
            cell_types.append(cell_type)
    
    for cell_type in cell_types:
        input_filename = DATA_DIR_W_CHROM + cell_type + "-" + mod1 + DATA_EXTENSION
        inputs.append(extract_wig_gz(input_filename, split = split))
        output_filename = DATA_DIR_W_CHROM + cell_type + "-" + mod2 + DATA_EXTENSION
        outputs.append(extract_wig_gz(output_filename, split = split))
    
    return np.array(inputs), np.array(outputs)
    
def process_data(split = None):
    """
    Processes all data in CONVERTEDDATADIR according to split. Saves data as .npy
    files in specified directory.
    
    data originally has 1925196 numbers per sample
    """
    save_dir = "../data/processed_data_" + str(split) + "/"
    for file in os.listdir(DATA_DIR):
        if file == ".ds_Store":
            continue
        filename = DATA_DIR + file
        splitted = extract_wig_gz(filename, split = split)
        np.save(save_dir + file, splitted)

#==============================================================================
# x = time.time()
# mod1 = "H3K27me3"
# mod2 = "H3K36me3"
# a = load_training_data(mod1, mod2)
# print(a)
# print(time.time() - x)
#==============================================================================



#process_data(192519)

if __name__ == "__main__":
    a = load_training_data("H3K27me3", "H3K36me3", split = self.img_cols)
    a = list(a)

