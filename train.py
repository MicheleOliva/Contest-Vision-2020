import os
import pickle
import sys

#### PATH LIBRERIE #######################################################################
#sys.path.insert(0, "/content/Contest-Vision-2020/pymodules/dataset_utils")
##########################################################################################

from datetime import datetime
from tensorflow.keras.applications import MobileNetV3Large
from keras import Model, Sequential
from keras.utils import plot_model
from keras.models import load_model
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.metrics import MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau, TensorBoard
from dataset_utils.data_augmentation import CustomAugmenter
from dataset_utils.data_generation import DataGenerator
from dataset_utils.data_loading import CustomDataLoader
from dataset_utils.preprocessing import CustomPreprocessor
from dataset_utils.output_encoding import CustomOutputEncoder
from age_estimation_utils.custom_metrics import rounded_mae
import argparse
import re
from keras import backend as K

# Argomenti da riga di comando
argparser = argparse.ArgumentParser(description='Starts training in a generated directory. With --resume resumes training from the last saved model in the current working directory.')
argparser.add_argument('--resume', action='store_true')
resume = argparser.parse_args().resume

datetime = datetime.today().strftime('%Y%m%d_%H%M%S')


# Modello

# Se si riprende il training
if resume:
  print("Resuming training from latest model...")
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
  initial_epoch = last_model_epochs


# Se non si riprende il training
else:
  # Creazione del modello
  print("Creating new model...")

  #### Parametri modello ###################################################################
  classifier_activation = 'relu'
  weights = 'imagenet'
  classes = 1
  input_shape = (96, 96, 3)
  alpha = 1.0
  ##########################################################################################

  # Utilizziamo la rete preallenata su imagenet, ma gli ultimi livelli vengono generati ad-hoc dalla versione senza pesi
  m1 = MobileNetV3Large(input_shape=input_shape, alpha=alpha, weights=weights, include_top=False)
  m2 = MobileNetV3Large(input_shape=input_shape, alpha=alpha, classes=classes, weights=None, classifier_activation=classifier_activation)

  # Per cui prendiamo il top del modello non preallenato e lo mettiamo come top del modello preallenato
  model = Sequential()
  model.add(m1)
  for i in range(-6, 0):
    layer = m2.get_layer(index=i)
    model.add(layer)

  m1 = None
  m2 = None

  # Creiamo una cartella per il training di questo modello e facciamo cd 
  MODEL_NAME = f"MNV3L_{input_shape[0]}x{input_shape[1]}_c{classes}_a{alpha}_{weights}"
  dirnm = f"{datetime}_{MODEL_NAME}"
  if not os.path.isdir(dirnm): 
    os.mkdir(dirnm)
  os.chdir(dirnm)

  # Compiliamo il modello
  optimizer = Adam()
  loss = MeanSquaredError()
  metrics = [MeanAbsoluteError(name='mae'),
             rounded_mae]

  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  initial_epoch = 0

  
model.summary()
input_shape = tuple(model.get_layer(index=0).inputs[0].shape[1:])

# Training

# Generatore del training set

#### Parametri generatore ################################################################
train_csv_path = '/content/drive/Shareddrives/Progettone/Age Estimation/caches/train_cvs.cache' # METTI DUMP .cache
train_dataset_root_path = '/content/train'
train_batch_size = 32
train_epoch_mode = 'identities' # Len del generator è il numero di identities
corruptions_prob = 0.5
frequent_corruptions_prob = 0.85
epoch_multiplier = 50
##########################################################################################

train_preprocessor = CustomPreprocessor(desired_shape=input_shape[:2])
train_augmenter = CustomAugmenter(corruptions_prob, frequent_corruptions_prob)
train_encoder = CustomOutputEncoder()
train_loader = CustomDataLoader(mode='training', csv_path=train_csv_path, csv_names=None, dataset_root_path=train_dataset_root_path, batch_size=train_batch_size)

train_generator = DataGenerator(mode='training', preprocessor=train_preprocessor, data_augmenter=train_augmenter, output_encoder=train_encoder, data_loader=train_loader, batch_size=train_batch_size, epoch_mode=train_epoch_mode, epoch_multiplier=epoch_multiplier)

# Generatori per l'evaluation

#### Parametri generatore #################################################################
eval_csv_path = '/content/drive/Shareddrives/Progettone/Age Estimation/caches/eval_csv.cache' # METTI DUMP .cache
eval_dataset_root_path = '/content/eval'
eval_batch_size = 1
eval_epoch_mode = 'identities' # Len del generator è il numero di identities
###########################################################################################

eval_preprocessor = CustomPreprocessor(desired_shape=input_shape[:2])
eval_encoder = CustomOutputEncoder()
eval_loader = CustomDataLoader(mode='validation', csv_path=eval_csv_path, csv_names=None, dataset_root_path=eval_dataset_root_path, batch_size=eval_batch_size)

eval_generator = DataGenerator(mode='validation', preprocessor=eval_preprocessor, data_augmenter=None, output_encoder=eval_encoder, data_loader=eval_loader, batch_size=eval_batch_size, epoch_mode=eval_epoch_mode, epoch_multiplier=epoch_multiplier)


## Callbacks

#### Parametri per le callback ###########################################################
path = "."
min_delta = 0.01 # Quanto deve scendere il mae per esser considerato migliorato
checkpoint_monitor = 'val_mae' # Solo per il model_checkpoint
monitor = 'val_loss'
mode = 'auto' # Controllare che funzioni, ossia il mae deve scendere per essere considerato migliorato
factor = 0.2 # lr = lr * factor
patience_lr = 5 # Cambiare in base alla lunghezza dell'epoca
# patience_stop = 5
checkpoint_path = os.path.join(path, "epoch{epoch:02d}_mae{val_mae:.2f}.model")
save_best_only = False
logdir = os.path.join(path, "tensorboard")
##########################################################################################

if not os.path.isdir(logdir): 
  os.mkdir(logdir)

append = os.path.exists("training_log.csv")
logger = CSVLogger(os.path.join(path, f"training_log.csv"), append=append)

reduce_lr_plateau= ReduceLROnPlateau(monitor=monitor, 
                                     factor=factor, 
                                     mode=mode, 
                                     verbose=1, 
                                     patience=patience_lr, 
                                     cooldown=1, 
                                     min_delta=min_delta)

#early_stopping = EarlyStopping(patience=patience_stop, 
#                               verbose=1, 
#                               restore_best_weights=True, 
#                               monitor=monitor, 
#                               mode=mode, 
#                               min_delta=min_delta)

model_checkpoint = ModelCheckpoint(checkpoint_path, 
                                   verbose=1, 
                                   save_weights_only=False,
                                   save_best_only=save_best_only, 
                                   monitor=checkpoint_monitor, 
                                   mode=mode)

tensorboard = TensorBoard(log_dir=logdir, 
                          write_graph=True, 
                          write_images=True)


# Fit del modello

#### Parametri di training ###############################################################
training_epochs = 10000
# Settare a true se si vuole cambiare learning rate
override_lr = False
new_lr_value = None # settare al nuovo valore desiderato del learning rate
##########################################################################################

if override_lr:
    K.set_value(model.optimizer.lr, new_lr_value)

# Print output to stdout in addition to console: command | tee -a /path/to/file ('command' is the command you use to run this script)
history = model.fit(train_generator, 
                    validation_data=eval_generator, 
                    initial_epoch=initial_epoch,
                    epochs=training_epochs, 
                    callbacks=[model_checkpoint, 
                               logger, 
                               reduce_lr_plateau,
                               tensorboard],
                    use_multiprocessing=False,
                    shuffle=False)


# Saving last model
model.save(os.path.join(path, f"{datetime}_final.model"))

# Saving history
with open(os.path.join(path, f"{datetime}_training_history"), 'wb') as history_file:
  print("Saving history.")
  pickle.dump(history, history_file)