#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import scipy.misc

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def one_hot(size):
    y = np.zeros(size)
    idx = np.random.randint(size[1], size=[size[0]])
    for b, c in enumerate(idx):
        y[b,c] = 1
    return y

def batch_norm(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

