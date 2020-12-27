import tensorflow
from data_loading import CustomDataLoader
from preprocessing import CustomPreprocessor
from data_augmentation import CustomAugmenter
from output_encoding import CustomOutputEncoder
from math import floor
import numpy as np


"""
    Strategy design pattern w.r.t. pre-processing, data augmentation, output enconding and file loading
"""
class DataGenerator(tensorflow.keras.utils.Sequence):

    def __init__(self, mode, preprocessor: CustomPreprocessor, data_augmenter: CustomAugmenter, output_encoder: CustomOutputEncoder, data_loader: CustomDataLoader, batch_size, epoch_mode = 'full'):
        if data_loader is None or batch_size is None:
            raise TypeError('Data generator needs data loader and batch size to be specified')

        self.allowed_modes = ['training', 'validation', 'testing']
        if mode is None or mode not in self.allowed_modes:
            raise TypeError("'mode' must be one of ['training', 'validation', 'testing']")

        self.allowed_epoch_modes = ['full', 'identities']
        if epoch_mode is None or epoch_mode not in self.allowed_epoch_modes:
            raise TypeError("'epoch_mode' must be one of ['full', 'identities']")

        self.mode = mode
        self.epoch_mode = epoch_mode
        self.preprocessor = preprocessor
        self.data_augmenter = data_augmenter
        self.output_encoder = output_encoder
        self.data_loader = data_loader
        self.batch_size = batch_size
    
    def __len__(self):
        if self.epoch_mode == 'full':
            return floor(self.data_loader.get_num_samples()/self.batch_size)
        else:
            # 'epoch_mode' is 'identities':
            return floor(min(self.data_loader.get_num_identities()*50, self.data_loader.get_num_samples())/self.batch_size)

    # Consider putting a lock as a class member and using it when modifying indexes (useful with multi-threading)
    # curr_batch è il numero del batch per il quale ci stanno chiedendo i sample
    def __getitem__(self, curr_batch):
        # x_batch and y_batch cannot be numpy arrays since images do not have the same size
        x_batch, y_batch, roi_batch = self.data_loader.load_batch(curr_batch)

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
            raise NotImplementedError('Testing mode is not implemented yet')
            #return x_batch
    
    def on_epoch_end(self):
        if self.mode == 'training':
            self.data_loader.epoch_ended()
