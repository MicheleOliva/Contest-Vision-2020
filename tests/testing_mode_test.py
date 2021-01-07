
from keras.models import load_model

model = load_model('')

print('Loaded model!')


from dataset_utils.data_loading import CustomDataLoader
from dataset_utils.data_generation import DataGenerator


BATCH_SIZE = 1
data_loader = CustomDataLoader(mode='testing', csv_names=None, csv_path='', dataset_root_path='', batch_size=BATCH_SIZE)

from dataset_utils.preprocessing import CustomPreprocessor
preprocessor = CustomPreprocessor(desired_shape=(96,96))
data_generator = DataGenerator(preprocessor=preprocessor, data_augmenter=None, output_encoder=None, mode='testing', data_loader=data_loader, batch_size=BATCH_SIZE)


"""
import cv2
import numpy as np

length = len(data_generator)
print('length: ', length)
for i in range(0, length):
    img = data_generator[i]
    img = img.astype(np.uint8)
    cv2.imshow('image', img[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""


predictions = model.predict(data_generator)

print('Lunghezza predictions: ', len(predictions))
print('Predictions: ', predictions)
