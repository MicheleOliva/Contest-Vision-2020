from data_loading import CustomDataLoader
from data_generation import DataGenerator


BATCH_SIZE = 32
MODE = 'training'
CSV_PATH = ''
DATASET_ROOT_PATH = ''
data_loader = CustomDataLoader(mode=MODE, csv_names=None, csv_path=CSV_PATH, dataset_root_path=DATASET_ROOT_PATH, batch_size=BATCH_SIZE)

from preprocessing import CustomPreprocessor
preprocessor = CustomPreprocessor(desired_shape=(96,96))
data_generator = DataGenerator(preprocessor=None, data_augmenter=None, output_encoder=None, mode=MODE, data_loader=data_loader, batch_size=BATCH_SIZE)

length = len(data_generator)
print(f'Found {length} batches')
for i in range(0, length):
    item = data_generator.__getitem__(i)
print('\n')
data_generator.on_epoch_end()

# Secondo round
length = len(data_generator)
print(f'Found {length} batches')
for i in range(0, length):
    item = data_generator.__getitem__(i)
print('\n')
data_generator.on_epoch_end()
