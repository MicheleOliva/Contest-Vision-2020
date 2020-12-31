import pandas as pd
import random
import cv2
import numpy as np
import pickle
import os

class VggDataLoader():
    def __init__(self, mode, csv_path, csv_names, dataset_root_path, batch_size, csv_sep=','):
        """
            mode is one of 'training', 'validation', 'testing'

            Just pass None as csv_names if you don't need to load custom header names. The same holds for csv_sep.
        """
        self.allowed_modes = ['training', 'validation', 'testing']
        if mode is None or mode not in self.allowed_modes:
            raise TypeError("'mode' must be one of ['training', 'validation', 'testing']")

        if csv_path is None:
            raise TypeError('Please provide a .csv annotations file!')

        if dataset_root_path is None or batch_size is None or csv_sep is None:
            raise TypeError("'dataset_root_path', 'batch_size' and 'csv_sep' cannot be None")

        self.batch_size = batch_size
        self.mode = mode
        self.dataset_root_path = dataset_root_path

        self.identities = [] # list to hold all of the ids, in order to be able to build balanced batches
        self.identities_persistent = []

        self.groundtruth_metadata = {} 

        self.test_data = []

        self.num_samples = 0

        self.images = []
        self.labels = []
        self.rois = []
        
        if self.mode != 'testing':
            self._init_training_validation(csv_path, csv_sep, csv_names)
        else:
            self._init_testing(csv_path, csv_sep, csv_names)
    
    def _init_training_validation(self, csv_path, csv_sep, csv_names):
        print('Loading training/validation data...')            
        if csv_path.split('.')[-1] == 'cache':
            # load cache
            # Assumes that the cache contains a list of all the identities, a dictionary containing metadata about those identities and the number of samples contained in the cache.
            # The dictionary must have the same format as the 'groundtruth_metadata' dictionary that is built below.
            # dati che mi servono: identities, groundtruth_metadata, num_samples
            with open(csv_path, 'rb') as cache_file:
                cache = pickle.load(cache_file)
            self.identities = cache['identities']
            self.identities_persistent = self.identities.copy()
            self.groundtruth_metadata = cache['groundtruth_metadata']
            self.num_samples = cache['num_samples']
        else:
            # Assumes for the provided csv the following structure:
            # Path, ID, Gender, Age, x_min(roi_origin_x), y_min(roi_origin_y), width(roi_width), height(roi_height)
            groundtruth = pd.read_csv(csv_path, sep=csv_sep, names=csv_names)
            # for each groundtruth row
            for gt_sample in groundtruth.iterrows():
                identity = gt_sample[1]["ID"]
                # this iteration is over all of the elements in groundtruth, so the same id can be encountered multiple times (same id associated to multiple images)
                if identity not in self.identities:
                    self.identities.append(identity)
                # load identity's metadata
                id_data = {
                    'age': gt_sample[1]["Age"],
                    'roi': {
                        'upper_left_x': gt_sample[1]["x_min"],
                        'upper_left_y': gt_sample[1]["y_min"],
                        'width': gt_sample[1]["width"],
                        'height': gt_sample[1]["height"]
                    },
                    'path': gt_sample[1]["Path"]
                }
                if identity not in self.groundtruth_metadata.keys():
                    self.groundtruth_metadata[identity] = {
                        'index': 0,
                        'metadata': []
                    }
                # the other elements in the list associated to an identity are metadata 
                self.groundtruth_metadata[identity]['metadata'].append(id_data)
                self.num_samples += 1
            # Dump loaded data to cache
            # Split csv path in directory path and filename
            (csv_dir, csv_name) = os.path.split(csv_path)
            # Create a name for cache file with the same name as csv file but different extension
            cache_name = csv_name.split('.')[0]+'.cache'
            # Create a path pointing to the new cache file, locating it in the same directory as the csv file
            cache_path = os.path.join(csv_dir, cache_name)
            # Write relevant data to file
            with open(cache_path, 'wb') as cache_out_file:
                out_dict = {}
                out_dict['identities'] = self.identities
                out_dict['groundtruth_metadata'] = self.groundtruth_metadata
                out_dict['num_samples'] = self.num_samples
                pickle.dump(out_dict, cache_out_file)
            self.identities_persistent = self.identities.copy()   
        print('Finished loading training/validation data!')
        if self.mode == 'training':
            self._shuffle()
            self._build_training_data_container()
        else:
            self._build_validation_data_container()

    def _build_training_data_container(self):
        print('Building training data batches')
        self.images = []
        self.labels = []
        self.rois = []
        num_identities = len(self.identities)
        num_ids_to_resample = 0
        reached_end = False
        multiplier = 0
        num_samples_processed = 0
        while not reached_end:
            # manage identities in a circular way 
            ids_start = (multiplier*self.batch_size)%num_identities # identities' batch start
            ids_end = ((multiplier+1)*self.batch_size)%num_identities # identities' batch end
            # Manage the indetities array in a circular manner
            #batch_identities = self.identities[ids_start:ids_end] if ids_start < ids_end else self.identities[ids_start:].append(self.identities[:ids_end])
            if ids_start < ids_end:
                batch_identities = self.identities[ids_start:ids_end]
            else:
                batch_identities = self.identities[ids_start:]
                batch_identities.extend(self.identities[:ids_end])

            for identity in batch_identities:
                print(f'Processed {num_samples_processed} samples', end='\r')
                identity_data = self.groundtruth_metadata[identity]
                # if there are images available for that identity
                if identity_data['index'] < len(identity_data['metadata']):
                    # read the image and the necessary metadata
                    img_info = identity_data['metadata'][identity_data['index']]
                    img_path = os.path.join(self.dataset_root_path, img_info['path'])
                    self.images.append(img_path)
                    self.labels.append(img_info['age'])
                    self.rois.append(img_info['roi'])
                    # increase the index, because another sample for that identity has been used
                    identity_data['index'] += 1
                    num_samples_processed += 1
                else:
                    num_ids_to_resample += 1
                    removed_index = self.identities.index(identity)
                    self.identities.remove(identity)
                    # ids_end adjustement is needed by subsequent while loop
                    if removed_index < ids_end:
                        ids_end -= 1
                    elif removed_index == ids_end and ids_end == num_identities-1:
                        ids_end = 0
                    num_identities -= 1

            # if for some identities there weren't available images, take them from other identities
            # note that this mechanism solves also the problems arising when less than batch_size identities are available, by
            # picking multiple images from the available entities
            # the __len__ method in the data generator associated to this data loader is responsible for avoiding that this
            # method is called when less than batch_size "fresh" images are available
            while(num_ids_to_resample > 0 and num_identities > 0):
                print(f'Processed {num_samples_processed} samples', end='\r')
                identity = self.identities[ids_end] # remeber that slicing at previous step excludes upper limit
                identity_data = self.groundtruth_metadata[identity]
                if identity_data['index'] < len(identity_data['metadata']):
                    # read the image and the necessary metadata
                    img_info = identity_data['metadata'][identity_data['index']]
                    img_path = os.path.join(self.dataset_root_path, img_info['path'])
                    self.images.append(img_path)
                    self.labels.append(img_info['age'])
                    self.rois.append(img_info['roi'])

                    num_ids_to_resample -= 1
                    identity_data['index'] += 1
                    num_samples_processed += 1
                else:
                    self.identities.remove(identity)
                    num_identities -= 1
                    
                ids_end = ((ids_end+1)%num_identities)
            if num_identities == 0:
                reached_end = True
            multiplier += 1
        print('\nFinished building training data batches')
    
    def _build_validation_data_container(self):
        for identity in self.identities:
            for data_sample in self.groundtruth_metadata[identity]['metadata']:
                img_path = os.path.join(self.dataset_root_path, data_sample['path'])
                self.images.append(img_path)
                self.labels.append(data_sample['age'])
                self.rois.append(data_sample['roi'])
        
    def _init_testing(self, csv_path, csv_sep, csv_names):
        self.test_data = []
        # mode is surely 'testing'
        # load metadata from csv (need for face bounding boxes)
        # Assumes for the provided csv the following structure:
        # frame_origin_x (0 for VGGFace), frame_origin_y (0 for VGGFace), path, id, roi_origin_x, roi_origin_y, roi_width, roi_height
        test_data = pd.read_csv(csv_path, sep=csv_sep, names=csv_names)
        print('Loading test data...')
        for test_sample in test_data.iterrows():
            data = {
                test_sample[1]['Path']: {
                    'roi_origin_x': test_sample[1]['x_min'],
                    'roi_origin_y': test_sample[1]['y_min'],
                    'roi_width': test_sample[1]['width'],
                    'roi_height': test_sample[1]['height']
                }
            }
            self.test_data.append(data)
            self.num_samples += 1
        print('Done loading test data!')

    def load_batch(self, batch_index):
        if self.num_samples < self.batch_size:
            raise ValueError('The number of samples is smaller than the batch size')
        
        if self.mode != 'testing':
            return self._yield_training_validation(batch_index)
        else:
            return self._yield_testing(batch_index)
        
    def _yield_training_validation(self, batch_index):
        images_to_yield_paths = self.images[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        images_to_yield = []
        labels_to_yield = self.labels[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        rois_to_yield = self.rois[batch_index*self.batch_size:(batch_index+1)*self.batch_size]

        for image_path in images_to_yield_paths:
            img = cv2.imread(image_path) # watch out for slashes (/)
            # if OpenCV is unable to read an image, it returns None
            if img is None:
                print('[DATA LOADER ERROR] cannot find image at path: ', image_path)
                path_index = images_to_yield_paths.index(image_path)
                del labels_to_yield[path_index]
                del rois_to_yield[path_index]
                continue
            img = img.astype('float32')
            images_to_yield.append(img)

        return images_to_yield, labels_to_yield, rois_to_yield

    def _yield_testing(self, batch_index):
        images_to_yield = []
        rois_to_yield = []
        data_to_yield = self.test_data[batch_index*self.batch_size:(batch_index+1)*self.batch_size]
        for sample in data_to_yield:
            # 'sample' has this structure:
            # {path: {
            #    'roi_origin_x': test_sample[1]['roi_origin_x'],
            #    'roi_origin_y': test_sample[1]['roi_origin_y'],
            #    'roi_width': test_sample[1]['roi_width'],
            #    'roi_height': test_sample[1]['roi_height'] 
            #   }      
            # }
            img_path = os.path.join(self.dataset_root_path, list(sample.keys())[0])
            img = cv2.imread(img_path) # watch out for slashes (/)
            # if the path does not exist or there are problems while reading the image
            if img is None:
                print('[DATA LOADER ERROR] cannot find image at path: ', img_path)
                continue
            roi_data = list(sample.values())[0]
            roi = {
                'upper_left_x': roi_data['roi_origin_x'],
                'upper_left_y': roi_data['roi_origin_y'],
                'width': roi_data['roi_width'],
                'height': roi_data['roi_height']
            }
            img = img.astype('float32')
            images_to_yield.append(img)
            rois_to_yield.append(roi)
        return images_to_yield, rois_to_yield

    def get_num_samples(self):
        return self.num_samples

    def get_num_identities(self):
        return len(self.identities)

    def epoch_ended(self):
        print('epoch ended')
        if self.mode == 'training':
            self.identities = self.identities_persistent.copy()
            self._shuffle(reinit_indexes=True)
            self._build_training_data_container()
        
    def _shuffle(self, reinit_indexes = False):
        print('Shuffling data...')
        # shuffle identities
        random.shuffle(self.identities)
        # shuffle images associated to each identity
        for identity in self.groundtruth_metadata.keys():
            random.shuffle(self.groundtruth_metadata[identity]['metadata'])
            if reinit_indexes:
                self.groundtruth_metadata[identity]['index'] = 0
        print('Finished shuffling data!')
    
    def _reinit_indexes(self):
        print('Reinitializing indexes...')
        for identity in self.groundtruth_metadata.keys():
            self.groundtruth_metadata[identity]['index'] = 0
        print('Indexes reinitialized!')