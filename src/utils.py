#!/usr/bin/env python

__author__ = "Meet Shah"
__license__ = "MIT"

import tensorflow as tf


def init_weights(shape, name):
    return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    
def init_biases(shape):
    return tf.Variable(tf.zeros(shape))


def batchNorm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


class batch_norm(object):
  	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
      		self.momentum = momentum
      		self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)


def threshold(x, val=0.5):
    x = tf.clip_by_value(x,0.5,0.5001) - 0.5
    x = tf.minimum(x * 10000,1) 
    return x

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

# def lrelu(x, leak=0.2):
#     f1 = 0.5 * (1 + leak)
#     f2 = 0.5 * (1 - leak)
#     return f1 * x + f2 * abs(x)