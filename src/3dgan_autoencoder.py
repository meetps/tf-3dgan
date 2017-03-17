#!/usr/bin/env python
import os
import sys

import numpy as np
import tensorflow as tf
import dataIO as d

from tqdm import *
from utils import *

'''
Global Parameters
'''
n_epochs   = 10000
n_ae_epochs= 1000
batch_size = 50
g_lr       = 0.0025
d_lr       = 0.00001
ae_lr      = 0.0001
beta       = 0.5
d_thresh   = 0.8 
z_size     = 200
leak_value = 0.2
cube_len   = 64
obj_ratio  = 0.5
reg_l2     = 0.001
gan_inter  = 50
ae_inter   = 50
obj        = 'chair' 

train_sample_directory = './train_sample/'
model_directory = './models/'
is_local = False

weights, biases = {}, {}

def generator(z, batch_size=batch_size, phase_train=True, reuse=False):

    strides    = [1,2,2,2,1]

    with tf.variable_scope("gen"):
        z = tf.reshape(z, (batch_size, 1, 1, 1, z_size))
        g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size,4,4,4,512), strides=[1,1,1,1,1], padding="VALID")
        g_1 = tf.nn.bias_add(g_1, biases['bg1'])                                  
        g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase_train)
        g_1 = tf.nn.relu(g_1)

        g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size,8,8,8,256), strides=strides, padding="SAME")
        g_2 = tf.nn.bias_add(g_2, biases['bg2'])
        g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
        g_2 = tf.nn.relu(g_2)

        g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size,16,16,16,128), strides=strides, padding="SAME")
        g_3 = tf.nn.bias_add(g_3, biases['bg3'])
        g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
        g_3 = tf.nn.relu(g_3)

        g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size,32,32,32,64), strides=strides, padding="SAME")
        g_4 = tf.nn.bias_add(g_4, biases['bg4'])
        g_4 = tf.contrib.layers.batch_norm(g_4, is_training=phase_train)
        g_4 = tf.nn.relu(g_4)
        
        g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'], (batch_size,64,64,64,1), strides=strides, padding="SAME")
        g_5 = tf.nn.bias_add(g_5, biases['bg5'])
        g_5 = tf.nn.sigmoid(g_5)

    print g_1, 'g1'
    print g_2, 'g2'
    print g_3, 'g3'
    print g_4, 'g4'
    print g_5, 'g5'
    
    return g_5

def encoder(inputs, phase_train=True, reuse=False):

    strides    = [1,2,2,2,1]
    with tf.variable_scope("dis"):
        d_1 = tf.nn.conv3d(inputs, weights['wd1'], strides=strides, padding="SAME")
        d_1 = tf.nn.bias_add(d_1, biases['bd1'])
        d_1 = tf.contrib.layers.batch_norm(d_1, is_training=phase_train)                               
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, weights['wd2'], strides=strides, padding="SAME") 
        d_2 = tf.nn.bias_add(d_2, biases['bd2'])
        d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        d_2 = lrelu(d_2, leak_value)
        
        d_3 = tf.nn.conv3d(d_2, weights['wd3'], strides=strides, padding="SAME")  
        d_3 = tf.nn.bias_add(d_3, biases['bd3'])
        d_3 = tf.contrib.layers.batch_norm(d_3, is_training=phase_train)
        d_3 = lrelu(d_3, leak_value) 

        d_4 = tf.nn.conv3d(d_3, weights['wd4'], strides=strides, padding="SAME")     
        d_4 = tf.nn.bias_add(d_4, biases['bd4'])
        d_4 = tf.contrib.layers.batch_norm(d_4, is_training=phase_train)
        d_4 = lrelu(d_4)

        d_5 = tf.nn.conv3d(d_4, weights['wae_d'], strides=[1,1,1,1,1], padding="VALID")     
        d_5 = tf.nn.bias_add(d_5, biases['bae_d'])
        d_5 = tf.nn.sigmoid(d_5)

    print d_5, 'ae5'

    return d_5

def discriminator(inputs, phase_train=True, reuse=False):

    strides    = [1,2,2,2,1]
    with tf.variable_scope("dis", reuse=True):
        d_1 = tf.nn.conv3d(inputs, weights['wd1'], strides=strides, padding="SAME")
        d_1 = tf.nn.bias_add(d_1, biases['bd1'])
        d_1 = tf.contrib.layers.batch_norm(d_1, is_training=phase_train)                               
        d_1 = lrelu(d_1, leak_value)

        d_2 = tf.nn.conv3d(d_1, weights['wd2'], strides=strides, padding="SAME") 
        d_2 = tf.nn.bias_add(d_2, biases['bd2'])
        d_2 = tf.contrib.layers.batch_norm(d_2, is_training=phase_train)
        d_2 = lrelu(d_2, leak_value)
        
        d_3 = tf.nn.conv3d(d_2, weights['wd3'], strides=strides, padding="SAME")  
        d_3 = tf.nn.bias_add(d_3, biases['bd3'])
        d_3 = tf.contrib.layers.batch_norm(d_3, is_training=phase_train)
        d_3 = lrelu(d_3, leak_value) 

        d_4 = tf.nn.conv3d(d_3, weights['wd4'], strides=strides, padding="SAME")     
        d_4 = tf.nn.bias_add(d_4, biases['bd4'])
        d_4 = tf.contrib.layers.batch_norm(d_4, is_training=phase_train)
        d_4 = lrelu(d_4)

        d_5 = tf.nn.conv3d(d_4, weights['wd5'], strides=[1,1,1,1,1], padding="VALID")     
        d_5 = tf.nn.bias_add(d_5, biases['bd5'])
        d_5 = tf.contrib.layers.batch_norm(d_5, is_training=phase_train)
        d_5 = tf.nn.sigmoid(d_5)

    print d_1, 'd1'
    print d_2, 'd2'
    print d_3, 'd3'
    print d_4, 'd4'
    print d_5, 'd5'

    return d_5

def initialiseWeights():

    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()

    weights['wg1'] = tf.get_variable("wg1", shape=[4, 4, 4, 512, 200], initializer=xavier_init)
    weights['wg2'] = tf.get_variable("wg2", shape=[4, 4, 4, 256, 512], initializer=xavier_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wg5'] = tf.get_variable("wg5", shape=[4, 4, 4, 1, 64], initializer=xavier_init)    

    weights['wd1'] = tf.get_variable("wd1", shape=[4, 4, 4, 1, 64], initializer=xavier_init)
    weights['wd2'] = tf.get_variable("wd2", shape=[4, 4, 4, 64, 128], initializer=xavier_init)
    weights['wd3'] = tf.get_variable("wd3", shape=[4, 4, 4, 128, 256], initializer=xavier_init)
    weights['wd4'] = tf.get_variable("wd4", shape=[4, 4, 4, 256, 512], initializer=xavier_init)    
    weights['wd5'] = tf.get_variable("wd5", shape=[4, 4, 4, 512, 1], initializer=xavier_init)    

    return weights

def initialiseBiases():
    
    global biases
    zero_init = tf.zeros_initializer()

    biases['bg1'] = tf.get_variable("bg1", shape=[512], initializer=zero_init)
    biases['bg2'] = tf.get_variable("bg2", shape=[256], initializer=zero_init)
    biases['bg3'] = tf.get_variable("bg3", shape=[128], initializer=zero_init)
    biases['bg4'] = tf.get_variable("bg4", shape=[64], initializer=zero_init)
    biases['bg5'] = tf.get_variable("bg5", shape=[1], initializer=zero_init)

    biases['bd1'] = tf.get_variable("bd1", shape=[64], initializer=zero_init)
    biases['bd2'] = tf.get_variable("bd2", shape=[128], initializer=zero_init)
    biases['bd3'] = tf.get_variable("bd3", shape=[256], initializer=zero_init)
    biases['bd4'] = tf.get_variable("bd4", shape=[512], initializer=zero_init)    
    biases['bd5'] = tf.get_variable("bd5", shape=[1], initializer=zero_init) 

    return biases

def trainGAN(is_dummy=False, exp_id=None):

    weights, biases =  initialiseWeights(), initialiseBiases()
    x_vector = tf.placeholder(shape=[batch_size,cube_len,cube_len,cube_len,1],dtype=tf.float32) 
    z_vector = tf.placeholder(shape=[batch_size,z_size],dtype=tf.float32) 

    # Weights for autoencoder pretraining
    xavier_init = tf.contrib.layers.xavier_initializer()
    zero_init = tf.zeros_initializer()
    weights['wae_d'] = tf.get_variable("wae_d", shape=[4, 4, 4, 512, 200], initializer=xavier_init)
    biases['bae_d'] =  tf.get_variable("bae_d", shape=[200], initializer=zero_init)

    encoded = encoder(x_vector, phase_train=True, reuse=False)
    encoded = tf.maximum(tf.minimum(encoded, 0.99), 0.01)
    decoded = generator(encoded, phase_train=True, reuse=False) 

    decoded_test = generator(tf.maximum(tf.minimum(encoder(x_vector, phase_train=False, reuse=False), 0.99), 0.01), phase_train=False, reuse=False)

    # Round decoder output
    decoded = threshold(decoded)
    # Compute MSE Loss and L2 Loss
    mse_loss = tf.reduce_mean(tf.pow(x_vector - decoded, 2))
    para_ae = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg', 'wd', 'wae'])]
    for var in tf.trainable_variables():
        if 'wd5' in var.name:
            last_layer_dis = var
    para_ae.remove(last_layer_dis)
    # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in para_ae])
    # ae_loss = mse_loss + reg_l2 * l2_loss
    ae_loss = mse_loss     

    optimizer_ae = tf.train.AdamOptimizer(learning_rate=ae_lr,beta1=beta, name="Adam_AE").minimize(ae_loss)
    # optimizer_ae = tf.train.RMSPropOptimizer(learning_rate=ae_lr, name="RMS_AE").minimize(ae_loss)


    net_g_train = generator(z_vector, phase_train=True, reuse=False) 

    d_output_x = discriminator(x_vector, phase_train=True, reuse=False)
    d_output_x = tf.maximum(tf.minimum(d_output_x, 0.99), 0.01)
    summary_d_x_hist = tf.summary.histogram("d_prob_x", d_output_x)

    d_output_z = discriminator(net_g_train, phase_train=True, reuse=True)
    d_output_z = tf.maximum(tf.minimum(d_output_z, 0.99), 0.01)
    summary_d_z_hist = tf.summary.histogram("d_prob_z", d_output_z)

    # Compute the discriminator accuracy
    n_p_x = tf.reduce_sum(tf.cast(d_output_x > 0.5, tf.int32))
    n_p_z = tf.reduce_sum(tf.cast(d_output_z <= 0.5, tf.int32))
    d_acc = tf.divide(n_p_x + n_p_z, 2 * batch_size)

    # Compute the discriminator and generator loss
    d_loss = -tf.reduce_mean(tf.log(d_output_x) + tf.log(1-d_output_z))
    g_loss = -tf.reduce_mean(tf.log(d_output_z))
    
    summary_d_loss = tf.summary.scalar("d_loss", d_loss)
    summary_g_loss = tf.summary.scalar("g_loss", g_loss)
    summary_n_p_z = tf.summary.scalar("n_p_z", n_p_z)
    summary_n_p_x = tf.summary.scalar("n_p_x", n_p_x)
    summary_d_acc = tf.summary.scalar("d_acc", d_acc)

    net_g_test = generator(z_vector, phase_train=False, reuse=True)

    para_g = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg', 'bg', 'gen'])]
    para_d = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wd', 'bd', 'dis'])]

    # only update the weights for the discriminator network
    optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr,beta1=beta).minimize(d_loss,var_list=para_d)
    # only update the weights for the generator network
    optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr,beta1=beta).minimize(g_loss,var_list=para_g)

    saver = tf.train.Saver(max_to_keep=50) 

    with tf.Session() as sess:  
      
        sess.run(tf.global_variables_initializer())        
        z_sample = np.random.normal(0, 0.33, size=[batch_size, z_size]).astype(np.float32)
        if is_dummy:
            volumes = np.random.randint(0,2,(batch_size,cube_len,cube_len,cube_len))
            print 'Using Dummy Data'
        else:
            volumes = d.getAll(obj=obj, train=True, is_local=is_local, obj_ratio=obj_ratio)
            print 'Using ' + obj + ' Data'
        volumes = volumes[...,np.newaxis].astype(np.float) 

        for epoch in range(n_ae_epochs):
            idx = np.random.randint(len(volumes), size=batch_size)
            x = volumes[idx]

            # Autoencoder pretraining
            # ae_l, mse_l, l2_l, _ = sess.run([ae_loss, mse_loss, l2_loss, optimizer_ae],feed_dict={x_vector:x})
            # print 'Autoencoder Training ', "epoch: ",epoch, 'ae_loss:', ae_l, 'mse_loss:', mse_l, 'l2_loss:', l2_l

            ae_l, mse_l, _ = sess.run([ae_loss, mse_loss, optimizer_ae],feed_dict={x_vector:x})
            print 'Autoencoder Training ', "epoch: ",epoch, 'ae_loss:', ae_l, 'mse_loss:', mse_l

            # output generated chairs
            if epoch % ae_inter == 10:
                idx = np.random.randint(len(volumes), size=batch_size)
                x = volumes[idx]
                decoded_chairs = sess.run(decoded_test, feed_dict={x_vector:x})
                if not os.path.exists(train_sample_directory):
                    os.makedirs(train_sample_directory)
                decoded_chairs.dump(train_sample_directory+'/ae_' + exp_id +str(epoch))

        for epoch in range(n_epochs):
            
            idx = np.random.randint(len(volumes), size=batch_size)
            x = volumes[idx]
            z = np.random.normal(0, 0.33, size=[batch_size, z_size]).astype(np.float32)

            # Update the discriminator and generator
            d_summary_merge = tf.summary.merge([summary_d_loss,
                                                summary_d_x_hist, 
                                                summary_d_z_hist,
                                                summary_n_p_x,
                                                summary_n_p_z,
                                                summary_d_acc])

            summary_d, discriminator_loss = sess.run([d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
            summary_g, generator_loss = sess.run([summary_g_loss,g_loss],feed_dict={z_vector:z})  
            d_accuracy, n_x, n_z = sess.run([d_acc, n_p_x, n_p_z],feed_dict={z_vector:z, x_vector:x})
            print n_x, n_z

            if d_accuracy < d_thresh:
                sess.run([optimizer_op_d],feed_dict={z_vector:z, x_vector:x})
                print 'Discriminator Training ', "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss, "d_acc: ", d_accuracy

            sess.run([optimizer_op_g],feed_dict={z_vector:z})
            print 'Generator Training ', "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss, "d_acc: ", d_accuracy

            # output generated chairs
            if epoch % gan_inter == 10:
                g_chairs = sess.run(net_g_test,feed_dict={z_vector:z_sample})
                if not os.path.exists(train_sample_directory):
                    os.makedirs(train_sample_directory)
                g_chairs.dump(train_sample_directory+'/'+str(epoch))
            
            if epoch % gan_inter == 10:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)      
                saver.save(sess, save_path = model_directory + '/' + str(epoch) + '.cptk')

if __name__ == '__main__':
    is_dummy = bool(int(sys.argv[1]))
    exp_id = sys.argv[2]
    trainGAN(is_dummy=is_dummy, exp_id=exp_id)
