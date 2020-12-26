import os
os.path.append("./pymodules/dataset_utils")
from data_augmenter import CustomAugmenter
from data_generation import DataGenerator
from data_loading import CustomDataLoader
from output_encoding import CustomOutputEncoder
from preprocessing import CustomPreprocessor
from PIL import Image

mode = 'training'
csv_path = ''
csv_names = ''
dataset_root_path = ''
batch_size = 1
csv_sep=','

preprocessor = CustomPreprocessor()
augmenter = CustomAugmenter()
encoder = CustomOutputEncoder()
loader = CustomDataLoader(mode=mode, csv_path=csv_path, csv_names=csv_names, dataset_root_path=dataset_root_path, batch_size=batch_size, csv_sep=csv_sep)

generator = DataGenerator(mode=mode, preprocessor=preprocessor, data_augmenter=augmenter, output_encoder=encoder, data_loader=loader, batch_size=batch_size)

for i in range(len(generator)):
    batch = DataGenerator[i]
    if i < 20:
        image = batch[1]
        image = Image.fromarray(image, mode='RGB')
        image.show()




