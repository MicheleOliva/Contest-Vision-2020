import h5py
import numpy as np
import os
import pickle
from tqdm import tqdm

base_path = './'  # dataset path
save_path = './train.hdf5'  # path to save the hdf5 file
dump_path = './train_dict.dump'

hf = h5py.File(save_path, 'a')  # open the file in append mode

identities = set()

with open(dump_path, 'rb') as dump:
    partition = pickle.load(dump)

dirs = list(partition.keys())
dirs.sort()

for k in tqdm(dirs):

    subgrp_name, img_name = os.path.split(k)
    if subgrp_name not in identities:
        subgrp = hf.create_group(subgrp_name)
        identities.add(subgrp_name)
    
    img_path = os.path.join(base_path, k)

    with open(img_path, 'rb') as img_f:  # open images as python binary
        binary_data = img_f.read()
    
    binary_data_np = np.asarray(binary_data)

    dset = subgrp.create_dataset(img_name, data=binary_data_np)


hf.close()
