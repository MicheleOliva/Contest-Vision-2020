import pandas as pd
import random
import cv2
import numpy as np
import pickle
import os


"""
    keras.preprocessing.image_data_from_directory
"""
class CustomDataLoader():
    """
    Takes care of loading data samples, groudntruth and eventual additional information from file.
    Is thought to be used as member of a DataGenerator.
    Assumes (meta)data regarding samples to build batch from is stored in a csv file.
    It build batches by loading image files as soon as the batch is requested, in order to minimize
    RAM usage.
    Batches contain images coming from different identities, until this is possible with the number 
    of images available.
    Note that this is a "stateful" implementation, so if for example the batch 0 has been requested, if it's requested
    again a different set of images is returned.
    Together with the images array, two other arrays, one containing labels and one containing rois,
    are built. The association between paths and labels and rois is positional, i.e the image in position 0 is
    associated to the label in position 0 and to the roi in position 0.
    """
    def __init__(self, mode, csv_path, csv_names, dataset_root_path, batch_size, csv_sep=','):
        """
        Parameters
        ----------
        mode : String 
            One of 'training', 'validation', 'testing'

        csv_names : String[]
            An array of strings to be used as column names when reading data from the given csv file.
            Just pass None if you don't need to load custom header names. 
        
        csv_sep : String
            Character(s) used to separate data in given csv file.
            Just pass None if you don't need to load custom header names. 
        """
        self.allowed_modes = ['training', 'validation', 'testing']
        if mode is None or mode not in self.allowed_modes:
            raise TypeError("'mode' must be one of ['training', 'validation', 'testing']")

        if csv_path is None and mode != 'testing':
            raise TypeError('Please provide groundtruth for training/validation data')

        if dataset_root_path is None or batch_size is None or csv_sep is None:
            raise TypeError("'dataset_root_path', 'batch_size' and 'csv_sep' cannot be None")

        self.batch_size = batch_size
        self.mode = mode
        self.dataset_root_path = dataset_root_path

        self.identities = [] # list to hold all of the ids, in order to be able to build balanced batches

        # dictionary to hold metadata about data samples associated to an identity, with the following structure:
        # {identity_id: {
        #   'index': index, 
        #   'metadata': [{},{},{}]
        # }}
        # where 'index' is the index of the next image (whose path will be taken from the 'metadata' list) related to indentity_id that must be included in a subsequent batch, and 'metadata'
        # is a list of dictionaries that contain each metadata regarding a specific data sample associated to the identity.
        self.groundtruth_metadata = {} 

        # List of dictionaries with the following structure:
        # {path: {
        #   'roi_origin_x': value,
        #   'roi_origin_y': value,
        #   'roi_width': value,
        #   'roi_height': value,
        # }}
        self.test_data = []

        self.num_samples = 0
        if self.mode != 'testing':
            self._init_train_valid(csv_path, csv_sep, csv_names)
        else:
            self._init_test(csv_path, csv_sep, csv_names)
    
    def _init_train_valid(self, csv_path, csv_sep, csv_names):
        """
        Build data structure used to build training/validation batches using information contained in given csv file.
        """
        # load groundtruth
        # last element following a dot is file's extension
        print('Loading data...')            
        if csv_path.split('.')[-1] == 'cache':
            # load cache
            # Assumes that the cache contains a list of all the identities, a dictionary containing metadata about those identities and the number of samples contained in the cache.
            # The dictionary must have the same format as the 'groundtruth_metadata' dictionary that is built below.
            # dati che mi servono: identities, groundtruth_metadata, num_samples
            with open(csv_path, 'rb') as cache_file:
                cache = pickle.load(cache_file)
            self.identities = cache['identities']
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
        print('Finished loading data!')
        if self.mode == 'training':
            self._shuffle()
    
    def _init_test(self, csv_path, csv_sep, csv_names):
        """
        Build data structure used to build test batches using information contained in given csv file.
        """
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
        """
        Call the appropriate functions to create the batch_index-th batch.
        """
        if self.num_samples < self.batch_size:
            raise ValueError('The number of samples is smaller than the batch size')
        
        if self.mode != 'testing':
            if len(self.identities) < self.batch_size:
                raise ValueError('The number of identities is smaller than the batch size')

        if self.mode != 'testing':
            return self._yield_training_validation(batch_index)
        else:
            return self._yield_testing(batch_index)
        
    def _yield_training_validation(self, batch_index):
        """
        Actually creates the batch_index-th batch of training/validation data. 
        The batch is composed of three arrays: one for images, one for labels and one for rois.
        The relation between paths and labels and rois is positional, i.e the image in position 0 is
        associated to the label in position 0 and to the roi in position 0.
        """
        # print(f'requested batch with index: {batch_index}') # DEBUG
        num_identities = len(self.identities)
        num_ids_to_resample = 0
        # manage identities in a circular way 
        ids_start = (batch_index*self.batch_size)%num_identities # identities' batch start
        ids_end = ((batch_index+1)*self.batch_size)%num_identities # identities' batch end
        # Manage the indetities array in a circular manner
        #batch_identities = self.identities[ids_start:ids_end] if ids_start < ids_end else self.identities[ids_start:].append(self.identities[:ids_end])
        if ids_start < ids_end:
            batch_identities = self.identities[ids_start:ids_end]
        else:
            batch_identities = self.identities[ids_start:]
            batch_identities.extend(self.identities[:ids_end])
        samples_batch = []
        labels_batch = []
        roi_batch = []
        for identity in batch_identities:
            identity_data = self.groundtruth_metadata[identity]
            # if there are images available for that identity
            if identity_data['index'] < len(identity_data['metadata']):
                # read the image and the necessary metadata
                img_info = identity_data['metadata'][identity_data['index']]
                img_path = os.path.join(self.dataset_root_path, img_info['path'])
                img = cv2.imread(img_path) # watch out for slashes (/)
                # if OpenCV is unable to read an image, it returns None
                if img is None:
                    print('[DATA LOADER ERROR] cannot find image at path: ', img_path)
                    # increase the index, in order to avoid this path when building subsequent batches with this identity
                    identity_data['index'] += 1
                    # sample another image from another identity to replace this one in the batch
                    num_ids_to_resample += 1
                    continue
                #batch.append(AgeEstimationSample(img, img_info['roi'], img_info['age'], 'BGR')) # cv2 reads as BGR
                img = img.astype('float32')
                samples_batch.append(img)
                labels_batch.append(img_info['age'])
                roi_batch.append(img_info['roi'])
                # increase the index, because another sample for that identity has been used
                identity_data['index'] += 1
            else:
                num_ids_to_resample += 1

        # if for some identities there weren't available images, take them from other identities
        # note that this mechanism solves also the problems arising when less than batch_size identities are available, by
        # picking multiple images from the available entities
        # the __len__ method in the data generator associated to this data loader is responsible for avoiding that this
        # method is called when less than batch_size "fresh" images are available
        last_taken_identity_index = ids_end 
        num_samples_when_last_taken = num_ids_to_resample
        while(num_ids_to_resample > 0):
            identity = self.identities[ids_end] # remeber that slicing at previous step excludes upper limit
            identity_data = self.groundtruth_metadata[identity]
            if identity_data['index'] < len(identity_data['metadata']):
                last_taken_identity_index = ids_end
                num_samples_when_last_taken = num_ids_to_resample
                # read the image and the necessary metadata
                img_info = identity_data['metadata'][identity_data['index']]
                img_path = os.path.join(self.dataset_root_path, img_info['path'])
                img = cv2.imread(img_path) # watch out for slashes (/)
                # if the path does not exist or there are problems while reading the image
                if img is None:
                    print('[DATA LOADER ERROR] cannot find image at path: ', img_path)
                    # increase the index, in order to avoid this path when building subsequent batches with this identity
                    identity_data['index'] += 1
                    continue
                #batch.append(AgeEstimationSample(img, img_info['roi'], img_info['age'], 'BGR')) # cv2 reads as BGR
                img = img.astype('float32')
                samples_batch.append(img)
                labels_batch.append(img_info['age'])
                roi_batch.append(img_info['roi'])

                num_ids_to_resample -= 1
                identity_data['index'] += 1
                
            ids_end = ((ids_end+1)%num_identities)
            if ids_end == last_taken_identity_index and num_ids_to_resample == num_samples_when_last_taken and identity_data['index'] == len(identity_data['metadata']):
                raise Exception(f'No more images available, missing {num_ids_to_resample} images!')

        # cannot return numpy arrays since images in batch have different sizes
        return samples_batch, labels_batch, roi_batch
        #return batch

    def _yield_testing(self, batch_index):
        """
        Actually creates the batch_index-th batch of test data.
        Testing mode doesn't work by identities, but works by samples.
        The batch is composed of two arrays: one for images and one for rois. 
        """
        samples_start = batch_index % self.num_samples
        samples_end = (batch_index+1) % self.num_samples
        if samples_start < samples_end:
            batch_samples = self.test_data[samples_start:samples_end]
        else:
            batch_samples = self.test_data[samples_start:]
            batch_samples.extend(self.test_data[:samples_end])
        images = []
        rois = []
        for sample in batch_samples:
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
            images.append(img)
            rois.append(roi)
        return images, rois
            
            
    def get_num_samples(self):
        return self.num_samples

    def get_num_identities(self):
        return len(self.identities)

    def epoch_ended(self):
        print('epoch ended')
        # if not really ended:
        #   return
        if self.mode == 'training':
            self._shuffle(reinit_indexes=True)
        elif self.mode == 'validation':
            self._reinit_indexes()

    def _shuffle(self, reinit_indexes = False):
        """
        Shuffles identities and data samples associated to each identity.
        so that the batches of the next epoch are different from the ones of the current epoch.
        """
        print('Shuffling data...')
        # set seed for reproducibility
        #random.seed()
        # shuffle identities
        random.shuffle(self.identities)
        # shuffle images associated to each identity
        for identity in self.groundtruth_metadata.keys():
            random.shuffle(self.groundtruth_metadata[identity]['metadata'])
            if reinit_indexes:
                self.groundtruth_metadata[identity]['index'] = 0
        print('Finished shuffling data!')
    
    def _reinit_indexes(self):
        """
        Reinitializes indexes used to keep track of which images have already been included in a batch,
        for each sample.
        """
        print('Reinitializing indexes...')
        for identity in self.groundtruth_metadata.keys():
            self.groundtruth_metadata[identity]['index'] = 0
        print('Indexes reinitialized!')