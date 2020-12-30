from keras.models import load_model
from tensorflow.keras.applications import MobileNetV3Large
from keras import Model, Sequential
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.metrics import MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau, TensorBoard
from data_augmentation import CustomAugmenter
from data_generation import DataGenerator
from data_loading import CustomDataLoader
from preprocessing import CustomPreprocessor

import os 

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

optimizer = Adam()
loss = MeanSquaredError()
metrics = [ MeanAbsoluteError(name='mae') ]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
initial_epoch = 0

input_shape = tuple(model.get_layer(index=0).inputs[0].shape[1:])

train_csv_path = os.path.abspath('') # METTI DUMP .cache
train_dataset_root_path = os.path.abspath('')
train_batch_size = 4
train_epoch_mode = 'full' # Len del generator è il numero di identities
corruptions_prob = 0.5
frequent_corruptions_prob = 0.85

train_preprocessor = CustomPreprocessor(desired_shape=input_shape[:2])
train_augmenter = CustomAugmenter(corruptions_prob, frequent_corruptions_prob)
train_loader = CustomDataLoader(mode='training', csv_path=train_csv_path, csv_names=None, dataset_root_path=train_dataset_root_path, batch_size=train_batch_size)

train_generator = DataGenerator(mode='training', preprocessor=train_preprocessor, data_augmenter=train_augmenter, output_encoder=None, data_loader=train_loader, batch_size=train_batch_size, epoch_mode=train_epoch_mode)

eval_csv_path = os.path.abspath('') # METTI DUMP .cache
eval_dataset_root_path = os.path.abspath('')
eval_batch_size = 1
eval_epoch_mode = 'full' # Len del generator è il numero di identities

eval_preprocessor = CustomPreprocessor(desired_shape=input_shape[:2])
eval_loader = CustomDataLoader(mode='validation', csv_path=eval_csv_path, csv_names=None, dataset_root_path=eval_dataset_root_path, batch_size=eval_batch_size)

eval_generator = DataGenerator(mode='validation', preprocessor=eval_preprocessor, data_augmenter=None, output_encoder=None, data_loader=eval_loader, batch_size=eval_batch_size, epoch_mode=eval_epoch_mode)

path = os.path.abspath('')
min_delta = 0.01 # Quanto deve scendere il mae per esser considerato migliorato
checkpoint_monitor = 'val_mae' # Solo per il model_checkpoint
monitor = 'val_loss'
mode = 'auto' # Controllare che funzioni, ossia il mae deve scendere per essere considerato migliorato
factor = 0.2 # lr = lr * factor
patience_lr = 3 # Cambiare in base alla lunghezza dell'epoca
patience_stop = 5
checkpoint_path = os.path.join(path, "epoch{epoch:02d}_mae{val_mae:.2f}.model")
logdir = os.path.join(path, "tensorboard")

logger = CSVLogger(os.path.join(path, "training_log.csv"), append=True)

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

training_epochs = 2

history = model.fit_generator(train_generator, 
                              validation_data=eval_generator, 
                              initial_epoch=initial_epoch,
                              epochs=training_epochs, 
                              callbacks=[model_checkpoint, 
                                         logger, 
                                         reduce_lr_plateau,
                                         early_stopping,
                                         tensorboard])
