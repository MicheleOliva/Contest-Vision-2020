import tensorflow as tf
from dataset_utils.groundtruth_encoding import OrdinalRegressionEncoder

def rounded_mae(y_true, y_pred):
  y_true = tf.round(y_true)
  y_pred = tf.round(y_pred)
  abs_difference = tf.abs(y_pred - y_true)
  return tf.reduce_mean(abs_difference, axis=-1)

def ordinal_rounded_mae(y_true, y_pred):
  y_true = tf.math.reduce_sum(y_true, axis=-1)
  print(f'y_true reduce sum: {y_true}')
  y_pred = tf.math.reduce_sum(y_pred, axis=-1)
  print(f'y_pred reduce sum: {y_pred}')
  abs_difference = tf.abs(y_pred - y_true)
  print(f'abs difference: {abs_difference}')
  return tf.reduce_mean(abs_difference, axis=-1)
  