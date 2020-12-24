import os
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from keras import Model, Sequential
from keras.utils import plot_model
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.metrics import MeanAbsoluteError
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau


# Creazione del modello
classifier_activation = 'relu'
weights = 'imagenet'
classes = 1
input_shape = (96, 96, 3)
alpha = 1.0

m1 = MobileNetV3Large(input_shape=input_shape, alpha=alpha, weights=weights, include_top=False)
m2 = MobileNetV3Large(input_shape=input_shape, alpha=alpha, classes=classes, weights=None, classifier_activation=classifier_activation)

i = Input([None, None, 3], dtype = tf.uint8)
x = tf.cast(i, tf.float32)
x = preprocess_input(x)
preprocess = Model(inputs=[i], outputs=[x], name="preprocessing")

model = Sequential()
model.add(preprocess)
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
train_generator = None
eval_generator = None
test_generator = None


## Parametri di training 
batch_size = 24
training_epochs = 100
initial_epoch = 0


## Creazione di una cartella
MODEL_NAME = f"MNV3L_{input_shape[0]}x{input_shape[1]}_c{classes}_a{alpha}_{weights}"
datetime = datetime.today().strftime('%Y%m%d_%H%M%S')
dirnm = f"{datetime}_{MODEL_NAME}"
if not os.path.isdir(dirnm): os.mkdir(dirnm)
path = os.path.join(".", dirnm)


## Callbacks
min_delta = 0.1 # Quanto deve scendere il mae per esser considerato migliorato
monitor = 'mae' # CONTROLLARE SE QUESTO NOME VA BENE!!!
mode = 'auto' # Controllare che funzioni, ossia il mae deve scendere per essere considerato migliorato
factor = 0.2 # lr = lr * factor
patience_lr = 2 # Cambiare in base alla lunghezza dell'epoca
patience_stop = 5

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

model_checkpoint = ModelCheckpoint(path, 
                                   verbose=1, 
                                   save_weights_only=False,
                                   save_best_only=True, 
                                   monitor=monitor, 
                                   mode=mode)


## Actual training
history = model.fit_generator(train_generator, 
                              validation_data=eval_generator, 
                              initial_epoch=initial_epoch,
                              epochs=training_epochs, 
                              callbacks=[model_checkpoint, 
                                         early_stopping, 
                                         logger, 
                                         reduce_lr_plateau])


## Saving history
with open(os.path.join(path, "training_history"), 'wb') as f:
  print("Saving history.")
  pickle.dump(history, f)


## Saving last model
model.save(os.path.join(path, f"{dirnm}_final"))