import sys
sys.path.insert(0, "./pymodules/dataset_utils")
from data_augmentation import CustomAugmenter
from data_generation import DataGenerator
from data_loading import CustomDataLoader
from output_encoding import CustomOutputEncoder
from preprocessing import CustomPreprocessor
import numpy as np
from google.colab.patches import cv2_imshow

mode = 'training'
csv_path = ''
csv_names = ''
dataset_root_path = ''
batch_size = 1
csv_sep=','

preprocessor = CustomPreprocessor(desired_shape=(96, 96))
augmenter = CustomAugmenter(0.5, 0.5)
encoder = CustomOutputEncoder()
loader = CustomDataLoader(mode=mode, csv_path=csv_path, csv_names=csv_names, dataset_root_path=dataset_root_path, batch_size=batch_size, csv_sep=csv_sep)

generator = DataGenerator(mode=mode, preprocessor=preprocessor, data_augmenter=augmenter, output_encoder=encoder, data_loader=loader, batch_size=batch_size)

VGGFACE2_MEANS = np.array([91.4953, 103.8827, 131.0912]) #BGR

for i in range(len(generator)):
    x_batch, y_batch = generator[i]
    if i < 20:
        image = x_batch[0]
        image = np.flip(image, 2)
        image = image + VGGFACE2_MEANS
        image = image.astype(np.uint8)
        cv2_imshow(image)
        print(f"Age: {y_batch[0]}")
        




