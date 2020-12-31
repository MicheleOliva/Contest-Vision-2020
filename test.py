import os
import sys

#### PATH LIBRERIE #######################################################################
#sys.path.insert(0, "/content/Contest-Vision-2020/pymodules/dataset_utils")
##########################################################################################

from keras.models import load_model
import argparse
import re
from dataset_utils.preprocessing import CustomPreprocessor
import cv2
import numpy as np
import csv


# Argomenti da riga di comando
argparser = argparse.ArgumentParser(description='Tests the model given given parameter.')
argparser.add_argument('csv', help='path to csv with images and roi annotations')
argparser.add_argument('dataset', help='path to the dataset')
argparser.add_argument('model', help='path to the model')
args = argparser.parse_args()

OUTPUT_PATH = 'predictions.csv'

# Generatore
def test_generator(csv_path, dataset_path, desired_shape):
    preprocessor = CustomPreprocessor(desired_shape=desired_shape)
    with open(csv_path, 'r') as csvfile:
        annotations = csv.reader(csvfile)
        for row in annotations:
            img = cv2.imread(os.path.join(dataset_path, row[2]))
            img = np.array(img)
            roi = np.array(row[4:], dtype='int')
            img = preprocessor.cut(img, roi)
            img = preprocessor.post_augmentation([img])
            yield np.array(img)



if not os.path.isdir(args.model):
    print("ERROR: model argument is not a directory")
    exit(0)

model = load_model(args.model, compile=True)
model.summary()
input_shape = tuple(model.get_layer(index=0).inputs[0].shape[1:])[:2]

pred = model.predict(test_generator(args.csv, args.dataset, input_shape), verbose=1)
pred = np.array(pred).reshape(-1) # Diventa un array lineare
pred = np.around(pred).astype('int') # arrotondamento per eccesso e difetto (1.1 diventa 1, 1.5 diventa 2)


# Salvataggio dell'output
with open(args.csv, 'r') as annotations_file:
    annotations = csv.reader(annotations_file)

    with open(OUTPUT_PATH, 'w') as output_file:
        output = csv.writer(output_file)
        
        for (row, age) in zip(annotations, pred):
            output.writerow([row[2], age])


        