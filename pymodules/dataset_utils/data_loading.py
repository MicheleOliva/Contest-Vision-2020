import pandas as pd
import random


"""
    keras.preprocessing.image_data_from_directory

"""
class CustomDataLoader():
    def __init__(self, mode, csv_path, csv_names, dataset_root_path, batch_size, csv_sep=','):
        """
            mode is one of 'training', 'validation', 'testing'

            Just pass None as csv_names if you don't need to load custom header names. The same holds for csv_sep.
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
        # load groundtruth
        # Assumes for the provided csv the following structure:
        # Path, ID, Gender, Age, x_min(roi_origin_x), y_min(roi_origin_y), width(roi_width), height(roi_height)
        groundtruth = pd.read_csv(csv_path, sep=csv_sep, names=csv_names)
        print('Loading data...')            
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
        print('Finished loading data!')
        if self.mode == 'training':
            self._shuffle()
    
    def _init_test(self, csv_path, csv_sep, csv_names):
        # mode is surely 'testing'
        # load metadata from csv (need for face bounding boxes)
        # Assumes for the provided csv the following structure:
        # frame_origin_x (0 for VGGFace), frame_origin_y (0 for VGGFace), path, id, roi_origin_x, roi_origin_y, roi_width, roi_height
        test_data = pd.read_csv(csv_path, sep=csv_sep, names=csv_names)
        print('Loading test data...')
        for test_sample in test_data.iterrows():
            data = {
                test_sample[1]['path']: {
                    'roi_origin_x': test_sample[1]['roi_origin_x'],
                    'roi_origin_y': test_sample[1]['roi_origin_y'],
                    'roi_width': test_sample[1]['roi_width'],
                    'roi_height': test_sample[1]['roi_height']
                }
            }
            self.test_data.append(data)
            self.num_samples += 1
        print('Done loading test data!')

    def load_batch(self):
        if self.num_samples < self.batch_size:
            raise ValueError('The number of samples is smaller than the batch size')

        # TODO:
        # - considera batch_size identità che non hai considerato prima
        # - prendi un'immagine ciascuna
        # - per ogni immagine, metti l'immagine ed i relativi metadati in un oggetto AgeEstimationSample (file data_sample.py), e crea una lista con questi oggetti
        # - yielda la lista (vedi se serve altro oltre a yield per fare un iterator in python)
        # GESTIRE BENE situazioni del tipo non ho batch_size identità diverse, ma con quelle che ho apparo comunque a 64 immagini (farò qualche batch con più immagini
        # provenienti dalla stessa identità, ma pazienza)
        if self.mode != 'testing':
            self._yield_training_validation()
        else:
            self._yield_testing()
        
    def _yield_training_validation(self):
        batch_index = 0
        # manage identities in a circular way
        if 
        curr_batch_indentities = self.identities[batch_index : (batch_index+1)*self.batch_size]
        batch = []
        for identity in curr_batch_indentities:
            pass

    def _yield_testing(self):
        pass

    def get_num_samples(self):
        return self.num_samples

    def epoch_ended(self):
        if self.mode == 'training':
            self._shuffle(reinit_indexes=True)

    def _shuffle(self, reinit_indexes = False):
        print('Shuffling data...')
        # set seed for reproducibility
        random.seed(2020)
        # shuffle identities
        random.shuffle(self.identities)
        # shuffle images associated to each identity
        for identity in self.groundtruth_metadata.keys():
            random.shuffle(self.groundtruth_metadata[identity]['metadata'])
            if reinit_indexes:
                self.groundtruth_metadata[identity]['index'] = 0
        print('Finished shuffling data!')