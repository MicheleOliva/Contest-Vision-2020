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


'''
for i in os.listdir(base_path):   # read all the As'
    vid_name = os.path.join(base_path, i)
    grp = hf.create_group(vid_name)  # create a hdf5 group.  each group is one 'A'

    for j in os.listdir(vid_name):  # read all as' inside A
        track = os.path.join(vid_name, j)

        subgrp = grp.create_group(j)  # create a subgroup for the above created group. each small
                                      # a is one subgroup

        for k in os.listdir(track):   # find all images inside a.
            img_path = os.path.join(track, k)

            with open(img_path, 'rb') as img_f:  # open images as python binary
                binary_data = img_f.read()

            binary_data_np = np.asarray(binary_data)

            dset = subgrp.create_dataset(k, data=binary_data_np) # save it in the subgroup. each a-subgroup contains all the images.
'''

hf.close()
