import tensorflow
from dataset_utils.data_loading import CustomDataLoader
from dataset_utils.preprocessing import CustomPreprocessor
from dataset_utils.data_augmentation import CustomAugmenter
from dataset_utils.output_encoding import CustomOutputEncoder
from math import floor
import numpy as np


class DataGenerator(tensorflow.keras.utils.Sequence):
    """
    Implements a generator for the data to submit to the model during training/validation/testing process.
    
    Adopts a strategy design pattern w.r.t. pre-processing, data augmentation, output enconding and file loading,
    so that it's independent of the way such operations are performed.

    Supports two way to implement the concept of epoch completion:
    1) Full epoch: an epoch is considered completed when all the samples the data loader has access to have been
    explored.
    2) Epoch by identities: an epoch is considered completed when a sample for each of the identities of the dataset
    the data loader has access to has been explored. It's also possible to specify a multiplier so that the number of
    identities that have to be visited for the epoch to be considered completed it's equal to the number of the identities
    in the dataset multiplied by this multiplier.
    """
    def __init__(self, mode, preprocessor: CustomPreprocessor, data_augmenter: CustomAugmenter, output_encoder: CustomOutputEncoder, data_loader, batch_size, gt_encoder=None, epoch_mode='full', epoch_multiplier=50):
        """
        Parameters
        ----------
        mode : String
            The mode data has to be produced. Must be one of 'training', 'validation' or 'testing'.

        epoch_mode : String
            Which definition of epoch completion must be used. Must be one of 'full' or 'identities'.
        
        epoch_multiplier : Integer
            The multiplier to use with the number of identities in the dataset to consider an epoch completed.
        """
        if data_loader is None or batch_size is None:
            raise TypeError('Data generator needs data loader and batch size to be specified')

        self.allowed_modes = ['training', 'validation', 'testing']
        if mode is None or mode not in self.allowed_modes:
            raise TypeError("'mode' must be one of ['training', 'validation', 'testing']")

        self.allowed_epoch_modes = ['full', 'identities']
        if epoch_mode is None or epoch_mode not in self.allowed_epoch_modes:
            raise TypeError("'epoch_mode' must be one of ['full', 'identities']")

        if mode == 'testing' and epoch_mode != 'full':
            raise TypeError("Can't use 'testing' mode with epoch mode different from 'full'")
        
        if epoch_mode == 'identities' and epoch_multiplier <= 0:
            raise TypeError("When epoch_mode is 'identities' an integer epoch_multiplier >= 1 must be specified")

        self.mode = mode
        self.epoch_mode = epoch_mode
        self.preprocessor = preprocessor
        self.data_augmenter = data_augmenter
        self.output_encoder = output_encoder
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.gt_encoder = gt_encoder
        self.epoch_multiplier = epoch_multiplier
    
    def __len__(self):
        if self.epoch_mode == 'full':
            return floor(self.data_loader.get_num_samples()/self.batch_size)
        else:
            # 'epoch_mode' is 'identities':
            return floor(min(self.data_loader.get_num_identities()*self.epoch_multiplier, self.data_loader.get_num_samples())/self.batch_size)

    # Consider putting a lock as a class member and using it when modifying indexes (useful with multi-threading)
    # curr_batch è il numero del batch per il quale ci stanno chiedendo i sample
    def __getitem__(self, curr_batch):
        if curr_batch >= self.__len__():
            raise IndexError(f'Batch {curr_batch} is not available')

        if self.mode != 'testing':
            # x_batch and y_batch cannot be numpy arrays since images do not have the same size
            x_batch, y_batch, roi_batch = self.data_loader.load_batch(curr_batch)
        else:
            x_batch, roi_batch = self.data_loader.load_batch(curr_batch)
        
        if self.gt_encoder is not None:
            y_batch = self.gt_encoder.encode(y_batch)

        if self.preprocessor is not None:
            x_batch = self.preprocessor.pre_augmentation(x_batch, roi_batch)

        if self.data_augmenter is not None:
            self.data_augmenter.apply_augmentation(x_batch) # works in place

        if self.preprocessor is not None:
            x_batch = self.preprocessor.post_augmentation(x_batch)

        if self.mode != 'testing':
            if self.output_encoder is not None:
                self.output_encoder.encode(y_batch)
            # watch out when you create numpy arrays, 'cause Keras requires the batch dimension to be the first one
            return np.array(x_batch), np.array(y_batch) # automatically sets batch dimension as the first one
        else:
            #raise NotImplementedError('Testing mode is not implemented yet')
            return np.array(x_batch)
    
    def on_epoch_end(self):
        self.data_loader.epoch_ended()
