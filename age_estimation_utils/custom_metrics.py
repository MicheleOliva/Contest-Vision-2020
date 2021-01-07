import tensorflow as tf
from dataset_utils.groundtruth_encoding import OrdinalRegressionEncoder


def rounded_mae(y_true, y_pred):
 y_true = tf.round(y_true)
 y_pred = tf.round(y_pred)
 abs_difference = tf.abs(y_pred - y_true)
 return tf.reduce_mean(abs_difference, axis=-1)


class RoundedMae:
  
  def __init__(self):
    self.__name__ = "rounded_mae"

  def __call__(self, y_true, y_pred):
    y_true = tf.round(y_true)
    y_pred = tf.round(y_pred)
    abs_difference = tf.abs(y_pred - y_true)
    return tf.reduce_mean(abs_difference, axis=-1)
  
  def __str__(self):
    return self.__name__
