import tensorflow as tf

def rounded_mae(y_true, y_pred):
  y_true = tf.round(y_true)
  y_pred = tf.round(y_pred)
  abs_difference = tf.abs(y_pred - y_true)
  return tf.reduce_mean(abs_difference, axis=-1)
