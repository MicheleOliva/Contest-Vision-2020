import os
import sys

#### PATH LIBRERIE #######################################################################
sys.path.insert(0, "/content/Contest-Vision-2020/pymodules/dataset_utils")
##########################################################################################

from keras.models import load_model
import argparse
import re
from data_generation import DataGenerator
from vgg_data_loader import VggDataLoader
from output_encoding import CustomOutputEncoder
from preprocessing import CustomPreprocessor


# Argomenti da riga di comando
argparser = argparse.ArgumentParser(description='Evaluates the last saved model in the current working directory. You can specify a model to evaluate with --model.')
argparser.add_argument('-m', '--model', help='model path to evaluate')
args = argparser.parse_args()
# Modello

# Se si riprende il training
if args.model is None:
    print("Searching latest model...")
    # Cerchiamo il modello più recente in termini di epoche
    dir = os.listdir()

    # Cerchiamo l'ultima epoca nel nome
    regex = re.compile(r"epoch(\d+).*[.]model") 

    last_model = None
    last_model_epochs = 0

    # Dai file che corrispondono alla regex estraiamo il numero delle epoche
    # e scegliamo il modello con più epoche 
    for fname in dir:
        match = regex.match(fname)
        if match is not None and int(match[1]) >= last_model_epochs:
            last_model = fname
            last_model_epochs = int(match[1])

    if last_model is None:
        print("ERROR: could not find any model.")
        exit(0)

    print(f"Found {last_model}. Loading...")
    model = load_model(last_model, compile=True)

else:
    if not os.path.isdir(args.model):
        print("ERROR: model argument is not a directory")
        exit(0)
    model = load_model(args.model, compile=True)


model.summary()
input_shape = tuple(model.get_layer(index=0).inputs[0].shape[1:])

#### Parametri generatore #################################################################
eval_csv_path = '/content/drive/Shareddrives/Progettone/Age Estimation/caches/eval_csv.cache' # METTI DUMP .cache
eval_dataset_root_path = '/content/eval'
eval_batch_size = 1
eval_epoch_mode = 'full' # Len del generator è il numero di identities
###########################################################################################

eval_preprocessor = CustomPreprocessor(desired_shape=input_shape[:2])
eval_encoder = CustomOutputEncoder()
eval_loader = VggDataLoader(mode='validation', csv_path=eval_csv_path, csv_names=None, dataset_root_path=eval_dataset_root_path, batch_size=eval_batch_size)

eval_generator = DataGenerator(mode='validation', preprocessor=eval_preprocessor, data_augmenter=None, output_encoder=eval_encoder, data_loader=eval_loader, batch_size=eval_batch_size, epoch_mode=eval_epoch_mode)

model.evaluate(
    eval_generator,
    use_multiprocessing=False,
)