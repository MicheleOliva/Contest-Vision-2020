#!/usr/bin/env python3

import os
import sys
from keras.models import load_model
import re
from dataset_utils.preprocessing import CustomPreprocessor
from age_estimation_utils.custom_metrics import RoundedMae
import cv2
import numpy as np
import csv


# Argomenti da riga di comando

args_csv = "test.csv"
args_model = "" #INFILA
args_dataset = "" #INFILA 

OUTPUT_PATH = 'GROUP_02.csv'

# Generatore
def test_generator(csv_path, dataset_path, desired_shape):
    preprocessor = CustomPreprocessor(desired_shape=desired_shape)
    with open(csv_path, 'r') as csvfile:
        annotations = csv.reader(csvfile)
        for row in annotations:
            img_path = os.path.join(dataset_path, row[0])
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f'Unable to read image {img_path}')
            img = np.array(img)
            roi = np.array(row[1:], dtype='int')
            img = preprocessor.cut(img, roi)
            img = preprocessor.post_augmentation([img])
            yield np.array(img)


if not os.path.isdir(args_model):
    print(f"ERROR: model argument {args_model} is not a directory")
    exit(0)

rounded_mae = RoundedMae()
model = load_model(args_model, custom_objects={'rounded_mae':rounded_mae}, compile=True)
model.summary()
input_shape = tuple(model.get_layer(index=0).inputs[0].shape[1:])[:2]
print(f'Input shape: {input_shape}')

pred = model.predict(test_generator(args_csv, args_dataset, input_shape), verbose=1)
pred = np.array(pred).reshape(-1) # Diventa un array lineare
pred = np.around(pred).astype('int') # arrotondamento per eccesso e difetto (1.1 diventa 1, 1.5 diventa 2)


# Salvataggio dell'output
with open(args_csv, 'r') as annotations_file:
    annotations = csv.reader(annotations_file)

    with open(OUTPUT_PATH, 'w', newline='') as output_file:
        output = csv.writer(output_file)
        
        for (row, age) in zip(annotations, pred):
            output.writerow([row[0], age])


        