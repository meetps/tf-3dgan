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
strides    = [1,2,2,2,1]

def generator(z, batch_size=batch_size, phase_train=True, reuse=False):
 
    g_1 = tf.add(tf.matmul(z, weights['wg1']), biases['bg1'])
    g_1 = tf.reshape(g_1, [-1,4,4,4,512])
    g_1 = batchNorm(g_1, 512, phase_train)

    g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], output_shape=[batch_size,8,8,8,256], strides=strides, padding="SAME")
    g_2 = tf.nn.bias_add(g_2, biases['bg2'])
    g_2 = batchNorm(g_2, 256, phase_train)
    g_2 = tf.nn.relu(g_2)

    g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], output_shape=[batch_size,16,16,16,128], strides=strides, padding="SAME")
    g_3 = tf.nn.bias_add(g_3, biases['bg3'])
    g_3 = batchNorm(g_3, 128, phase_train)
    g_3 = tf.nn.relu(g_3)
    
    g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], output_shape=[batch_size,32,32,32,1], strides=strides, padding="SAME")
    g_4 = tf.nn.bias_add(g_4, biases['bg4'])                                   
    g_4 = tf.nn.sigmoid(g_4)
    
    return g_4


def discriminator(inputs, batch_size=batch_size, is_train=True, reuse=False):

    d_1 = tf.nn.conv3d(inputs, weights['wd1'], strides=strides, padding="SAME")
    d_1 = tf.nn.bias_add(d_1, biases['bd1'])                               
    d_1 = tf.nn.relu(d_1)

    d_2 = tf.nn.conv3d(d_1, weights['wd2'], strides=strides, padding="SAME") 
    d_2 = tf.nn.bias_add(d_2, biases['bd2'])                                  
    d_2 = batchNorm(d_2, 128, phase_train)
    d_2 = tf.nn.relu(d_2)
    
    d_3 = tf.nn.conv3d(d_2, weights['wd3'], strides=strides, padding="SAME") 
    d_3 = tf.nn.bias_add(d_3, biases['bd3'])                                  
    d_3 = batchNorm(d_3, 128, phase_train)
    d_3 = tf.nn.relu(d_3) 

    d_4 = tf.nn.conv3d(d_3, weights['wd4'], strides=strides, padding="SAME")     
    d_4 = tf.nn.bias_add(d_4, biases['bd4'])                              
    d_4 = batchNorm(d_4, 128, phase_train)
    d_4 = tf.nn.relu(d_4) 

    shape = d_4.get_shape().as_list()
    dim = numpy.prod(shape[1:])
    d_5 = tf.reshape(d_4, shape=[-1, dim])
    d_5 = tf.add(tf.matmul(z, weights['wg1']), biases['bd5'])
    
    return d_5

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
