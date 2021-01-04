import os
import re
import numpy as np
from dataset_utils.vgg_data_loader import VggDataLoader
from dataset_utils.data_generation import DataGenerator
from dataset_utils.preprocessing import CustomPreprocessor
from dataset_utils.groundtruth_encoding import OrdinalRegressionEncoder
from keras.models import load_model


def compute_classification_mae(y_true, y_pred):
    """Calcola il mae dagli output della classification. Assume che gli output della 
    classification siano con le probabilità, quindi gli uno e zero dobbiamo metterceli
    noi.

    Args:
        y_true (np.array): array di groundtruth nel formato [1,1,1,0,0,0,....]
        y_pred (np.array): array di prediction nel formato [0.1,0.5,0.8,0,0,0,....]
    """
    errors = []
    for gt_sample, pred_sample in zip(y_true, y_pred):
        predicted_age = np.where(pred_sample < 0.5)[0][0] - 1
        real_age = np.where(gt_sample == 0)[0][0] - 1
        error = abs(predicted_age - real_age)
        errors.append(error)
    errors_np = np.array(errors)
    return np.mean(errors_np)


EVAL_CSV_PATH = '/content/drive/Shareddrives/Progettone/Age Estimation/caches/eval_csv.cache'
EVAL_SET_PATH = '/content/eval'
MODE = 'validation'
BATCH_SIZE = 1
NUM_CLASSES = 101
EPOCH_MODE = 'full'


dir = os.listdir()

# Cerchiamo l'ultima epoca nel nome
regex = re.compile(r"epoch(\d+).*[.]model") 

last_model = None

# Dai file che corrispondono alla regex estraiamo il numero delle epoche
# e scegliamo il modello con più epoche 
for fname in dir:
    match = regex.match(fname)
    if match is not None:
        last_model = fname

if last_model is None:
    print("ERROR: could not find any model.")
    exit(0)

print(f"Found {last_model}. Loading...")
model = load_model(last_model, compile=True)

data_loader = VggDataLoader(MODE, EVAL_CSV_PATH, None, EVAL_SET_PATH, BATCH_SIZE)
preprocessor = CustomPreprocessor()
gt_encoder = OrdinalRegressionEncoder(NUM_CLASSES)

data_generator = DataGenerator(MODE, preprocessor, None, None, data_loader, BATCH_SIZE, gt_encoder, EPOCH_MODE)

print('Performing validation...')
gt_array = []
prediction_array = []
for i in range(0, len(data_generator)):
    if i % 1000 == 0:
      print(f'Processed {i} samples')

    x, y = data_generator[i]

    prediction = model.predict(x)

    gt_array.append(y)
    prediction_array.append(prediction)

print('\n')

rounded_mae = compute_classification_mae(gt_array, prediction_array)

print(f'MAE: {rounded_mae}')