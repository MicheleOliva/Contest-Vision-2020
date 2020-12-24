from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
import pickle
import numpy as np
os.chdir(BASE_DIR + MODEL_DIR)

batch_size = 24



training_epochs = 100
initial_epoch = 0

training_data_generator = DataGenerator(X1, X2, y, batch_size=batch_size)
validation_data_generator = DataGenerator(X1_val, X2_val, y_val, flip_classes=False)

logger_callback = CSVLogger('SENET50_multitask_training.log', append=False) #remember to put it to True if continuing a training

validation_min_delta = 0.1 / 100  # minimum variation of the monitored quantity for it to be considered improved
monitored_val_quantity = 'val_one_hot_categorical_accuracy'
reduce_lr_plateau_callback = ReduceLROnPlateau(monitor=monitored_val_quantity, mode='auto', verbose=1, patience=2, factor=0.2, cooldown=1, min_delta=validation_min_delta)

# Stops training entirely if, after 5 epochs, no improvement is found on the validation metric.
# This triggers after the previous callback if it fails to make the training effective again
stopping_callback = EarlyStopping(patience=5, verbose=1, restore_best_weights=True, monitor=monitored_val_quantity, mode='auto', min_delta=validation_min_delta)


save_callback = ModelCheckpoint("SENET50_multitask_training", verbose=1, save_weights_only=False,
                              save_best_only=True, monitor=monitored_val_quantity, mode='auto')
#raise Exception
history = SENET50_multitask.fit_generator(training_data_generator, validation_data=validation_data_generator, initial_epoch=initial_epoch,
                                   epochs=training_epochs, callbacks=[save_callback, stopping_callback, logger_callback, reduce_lr_plateau_callback], 
                                   use_multiprocessing=False)

with open('SENET50_multitask_training_history', 'wb') as f:
  print("Saving history.")
  pickle.dump(history, f)

SENET50_multitask.save('SENET50_multitask_completed_training')
