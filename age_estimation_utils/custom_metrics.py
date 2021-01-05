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



def ordinal_rounded_mae(y_true, y_pred):
  # CORREGGERE: devo convertire il vettore di probabilità che caccia la rete in un vettore di 0 e 1,
  # e l'età è data dall'indice del primo elemento che è zero, meno uno
  print('y_true: ', y_true[0][0])
  print('y_true > 0.5: ', y_true > 0.5)
  y_true = tf.math.reduce_sum(y_true, axis=-1)
  # print(f'y_true reduce sum: {y_true}')
  y_pred = tf.math.reduce_sum(y_pred, axis=-1)
  # print(f'y_pred reduce sum: {y_pred}')
  abs_difference = tf.abs(y_pred - y_true)
  # print(f'abs difference: {abs_difference}')
  return tf.reduce_mean(abs_difference, axis=-1)
  
# a = ordinal_rounded_mae([[1,1,0,0]], [[1,0,0,0]])
# print(a)