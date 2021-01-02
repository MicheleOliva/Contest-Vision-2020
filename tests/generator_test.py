from dataset_utils.data_loading import CustomDataLoader
from dataset_utils.data_generation import DataGenerator
from dataset_utils.vgg_data_loader import VggDataLoader
from dataset_utils.preprocessing import CustomPreprocessor
from dataset_utils.groundtruth_encoding import OrdinalRegressionEncoder
import os
import numpy as np
import cv2


BATCH_SIZE = 1 #32
MODE = 'validation'
CSV_PATH = os.path.abspath('')
DATASET_ROOT_PATH = os.path.abspath('')
data_loader = VggDataLoader(mode=MODE, csv_names=None, csv_path=CSV_PATH, dataset_root_path=DATASET_ROOT_PATH, batch_size=BATCH_SIZE)

preprocessor = CustomPreprocessor(desired_shape=(96,96))
gt_encoder = OrdinalRegressionEncoder(101)
data_generator = DataGenerator(preprocessor=preprocessor, gt_encoder=gt_encoder, data_augmenter=None, output_encoder=None, mode=MODE, data_loader=data_loader, batch_size=BATCH_SIZE)

""" x_batch, y_batch = data_generator[3]
for image,label in zip(x_batch, y_batch):
    image = image.astype(np.uint8)
    cv2.imshow('image', image)
    print(f'Label: {label}')
    cv2.waitKey(0)
cv2.destroyAllWindows() """


length = len(data_generator)
print(f'Found {length} batches')
for i in range(0, length):
    print(f'Requested batch with index {i}', end='\r')
    image, label = data_generator.__getitem__(i)
    image = image.astype(np.uint8)
    cv2.imshow('image', image[0])
    print(f'Label: {label}')
    cv2.waitKey(0)
cv2.destroyAllWindows()
print('\n')
data_generator.on_epoch_end()

# Secondo round
length = len(data_generator)
print(f'Found {length} batches')
for i in range(0, length):
    print(f'Requested batch with index {i}', end='\r')
    image, label = data_generator.__getitem__(i)
    image = image.astype(np.uint8)
    cv2.imshow('image', image[0])
    print(f'Label: {label}')
    cv2.waitKey(0)
cv2.destroyAllWindows()
print('\n')
data_generator.on_epoch_end()
