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
batch_size = 64 


def generator(z, batch_size=batch_size, phase_train=True, reuse=False):
 
    g_1 = tf.add(tf.matmul(z, weights['wd1']), biases['bd1'])
    g_1 = tf.reshape(g_1, [-1,4,4,4,512])
    g_1 = batchNorm(g_1, 512, phase_train)

    ## TODO : Add layers

    return g_1

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
