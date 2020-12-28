import os
import pickle
import sys
sys.path.insert(0, "/content/Contest-Vision-2020/pymodules/dataset_utils")
from datetime import datetime
from tensorflow.keras.applications import MobileNetV3Large
from keras import Model, Sequential
from keras.utils import plot_model
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.metrics import MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau, TensorBoard
from data_augmentation import CustomAugmenter
from data_generation import DataGenerator
from data_loading import CustomDataLoader
from output_encoding import CustomOutputEncoder
from preprocessing import CustomPreprocessor


# Creazione del modello
classifier_activation = 'relu'
weights = 'imagenet'
classes = 1
input_shape = (96, 96, 3)
alpha = 1.0

m1 = MobileNetV3Large(input_shape=input_shape, alpha=alpha, weights=weights, include_top=False)
m2 = MobileNetV3Large(input_shape=input_shape, alpha=alpha, classes=classes, weights=None, classifier_activation=classifier_activation)

model = Sequential()
model.add(m1)
for i in range(-6, 0):
  layer = m2.get_layer(index=i)
  model.add(layer)

model.summary()

m1 = None
m2 = None


# Compilazione del modello
optimizer = Adam()
loss = MeanSquaredError()
metrics = [ MeanAbsoluteError(name='mae') ]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


# Training

## Generatori

### Training
train_csv_path = '/content/drive/Shareddrives/Progettone/Age Estimation/caches/train_cvs.cache' # METTI DUMP .cache
train_dataset_root_path = '/content/train'
train_batch_size = 32
train_epoch_mode = 'identities' # Len del generator è il numero di identities
corruptions_prob = 0.5
frequent_corruptions_prob = 0.5

train_preprocessor = CustomPreprocessor(desired_shape=(96, 96))
train_augmenter = CustomAugmenter(corruptions_prob, frequent_corruptions_prob)
train_encoder = CustomOutputEncoder()
train_loader = CustomDataLoader(mode='training', csv_path=train_csv_path, csv_names=None, dataset_root_path=train_dataset_root_path, batch_size=train_batch_size)

train_generator = DataGenerator(mode='training', preprocessor=train_preprocessor, data_augmenter=train_augmenter, output_encoder=train_encoder, data_loader=train_loader, batch_size=train_batch_size, epoch_mode=train_epoch_mode)


### Evaluation
eval_csv_path = '/content/drive/Shareddrives/Progettone/Age Estimation/caches/eval_csv.cache' # METTI DUMP .cache
eval_dataset_root_path = '/content/eval'
eval_batch_size = 1
eval_epoch_mode = 'identities' # Len del generator è il numero di identities

eval_preprocessor = CustomPreprocessor(desired_shape=(96, 96))
eval_encoder = CustomOutputEncoder()
eval_loader = CustomDataLoader(mode='validation', csv_path=eval_csv_path, csv_names=None, dataset_root_path=eval_dataset_root_path, batch_size=eval_batch_size)

eval_generator = DataGenerator(mode='validation', preprocessor=eval_preprocessor, data_augmenter=None, output_encoder=eval_encoder, data_loader=eval_loader, batch_size=eval_batch_size, epoch_mode=eval_epoch_mode)


## Parametri di training
training_epochs = 10000
initial_epoch = 0


## Creazione di una cartella
MODEL_NAME = f"MNV3L_{input_shape[0]}x{input_shape[1]}_c{classes}_a{alpha}_{weights}"
datetime = datetime.today().strftime('%Y%m%d_%H%M%S')
dirnm = f"{datetime}_{MODEL_NAME}"
path = os.path.join(".", dirnm)
if not os.path.isdir(path): 
  os.mkdir(path)
os.chdir(path)
path = "." 

logdir = os.path.join(path, "tensorboard")
if not os.path.isdir(logdir): 
  os.mkdir(logdir)

## Callbacks
min_delta = 0.01 # Quanto deve scendere il mae per esser considerato migliorato
checkpoint_monitor = 'val_mae' # Solo per il model_checkpoint
monitor = 'val_loss'
mode = 'auto' # Controllare che funzioni, ossia il mae deve scendere per essere considerato migliorato
factor = 0.2 # lr = lr * factor
patience_lr = 5 # Cambiare in base alla lunghezza dell'epoca
patience_stop = 5
checkpoint_path = os.path.join(path, "epoch{epoch:02d}_mae{val_mae:.2f}.model")

logger = CSVLogger(os.path.join(path, "training_log.csv"), append=False)

reduce_lr_plateau= ReduceLROnPlateau(monitor=monitor, 
                                     factor=factor, 
                                     mode=mode, 
                                     verbose=1, 
                                     patience=patience_lr, 
                                     cooldown=1, 
                                     min_delta=min_delta)

early_stopping = EarlyStopping(patience=patience_stop, 
                               verbose=1, 
                               restore_best_weights=True, 
                               monitor=monitor, 
                               mode=mode, 
                               min_delta=min_delta)

model_checkpoint = ModelCheckpoint(checkpoint_path, 
                                   verbose=1, 
                                   save_weights_only=False,
                                   save_best_only=True, 
                                   monitor=checkpoint_monitor, 
                                   mode=mode)

tensorboard = TensorBoard(log_dir=logdir, 
                          write_graph=True, 
                          write_images=True)

## Actual training
history = model.fit_generator(train_generator, 
                              validation_data=eval_generator, 
                              initial_epoch=initial_epoch,
                              epochs=training_epochs, 
                              callbacks=[model_checkpoint, 
                                         logger, 
                                         reduce_lr_plateau,
                                         tensorboard])


## Saving last model
model.save(os.path.join(path, f"final.model"))

## Saving history
with open(os.path.join(path, "training_history"), 'wb') as history_file:
  print("Saving history.")
  pickle.dump(history, history_file)