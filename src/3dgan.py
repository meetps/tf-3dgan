#!/usr/bin/env python

__author__ = "Meet Shah"
__license__ = "MIT"

import os

import numpy as np
import tensorflow as tf


from utils import *

'''
Global Parameters
'''
n_epochs   = 20000
batch_size = 100
g_lr       = 0.0025
d_lr       = 0.00001
beta       = 0.5
alpha_1    = 5
alpha_2    = 0.0001
d_thresh   = 0.8    

def generator(z, batch_size=batch_size, phase_train=True, reuse=False):
 
    g_1 = tf.add(tf.matmul(z, weights['wd1']), biases['bd1'])
    g_1 = tf.reshape(g_1, [-1,4,4,4,512])
    g_1 = batchNorm(g_1, 512, phase_train)

    g_2 = tf.nn.conv3d_transpose(g_1, weights['wd2'], output_shape=[batch_size,8,8,8,256], strides=[1,2,2,2,1], padding="SAME")
    g_2 = tf.nn.bias_add(g_2, biases['bd2'])
    g_2 = batchNorm(g_2, 256, phase_train)
    g_2 = tf.nn.relu(g_2)

    g_3 = tf.nn.conv3d_transpose(g_2, weights['wd3'], output_shape=[batch_size,16,16,16,128], strides=[1,2,2,2,1], padding="SAME")
    g_3 = tf.nn.bias_add(g_3, biases['bd3'])
    g_3 = batchNorm(g_3, 128, phase_train)
    g_3 = tf.nn.relu(g_3)
    
    g_4 = tf.nn.conv3d_transpose(g_3, weights['wd4'], output_shape=[batch_size,32,32,32,1], strides=[1,2,2,2,1], padding="SAME")
    g_4 = tf.nn.bias_add(g_4, biases['bd4'])                                   
    g_4 = tf.nn.sigmoid(g_4)
    
    return g_4


def discriminator(inputs, batch_size=batch_size, is_train=True, reuse=False):
    ## TODO : Add layers
    pass

def trainGAN():
    ## TODO
    pass

def testGAN():
    ## TODO
    pass

def visualize():
    ## TODO
    pass

def saveModel():
    ## TODO
    pass
