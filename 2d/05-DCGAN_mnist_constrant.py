#!/usr/bin/env python2
"""
**05-DCGAN**

Usage Examples:
    python 05-DCGAN_mnist_constrant.py 
"""
import os
import sys

import numpy as np
import tensorflow as tf
import scipy.misc

from logger import logger
from utils.utils import *

from tensorflow.examples.tutorials.mnist import input_data 

'''
Global Parameters
'''
n_epochs   = 10000
batch_size = 128
g_lr       = 0.0002
d_lr       = 0.0002
beta       = 0.5
d_thresh   = 0.8
z_size     = 100
y_size     = 10
leak_value = 0.2

img_len    = 28
img_channel= 1

f_dim = 64

model_name = 'mnist_constrant_tfbn_softmax'
summary_dir = './summary/'
train_sample_directory = './train_sample/'
model_directory = './models/'

weights, biases = {}, {}

'''
Flags
'''
args = tf.app.flags
args.DEFINE_boolean('train', True, 'True for training, False for testing [%(default)s]')
args.DEFINE_boolean('dummy', False, 'True for random sampling as real input, False for cifar10 images as real input [%(default)s]')
args.DEFINE_string('name', model_name, 'Model name [%(default)s]')
args.DEFINE_string('ckpath', None, 'Checkpoint path for restoring (ex: models/mnist_asyn.ckpt-100) [%(default)s]')
# args.DEFINE_integer('f_dim', f_dim, 'Dimension of first layer in D [%(default)s]')
FLAGS = args.FLAGS


# --- [ DCGAN (Radford rt al., 2015)
def generator(z, y, batch_size=batch_size, phase_train=True, reuse=False):

    with tf.variable_scope("gen", reuse=reuse):
        # g_1 = tf.add(tf.matmul(z, weights['wg1']), biases['bg1'])
        yb = tf.reshape(y, [-1, 1, 1, y_size])
        z = tf.concat(axis=1, values=[z, y])

        g_1 = tf.matmul(z, weights['wg1'])
        g_1 = tf.layers.batch_normalization(g_1, training=phase_train, epsilon=1e-8)
        # g_1 = batch_norm(g_1)
        g_1 = tf.nn.relu(g_1)
        print ("G1: ", g_1)

        g_2 = tf.concat(axis=1, values=[g_1, y])
        g_2 = tf.matmul(g_2, weights['wg2'])
        # g_2 = tf.nn.bias_add(g_2, biases['bg2'])
        g_2 = tf.reshape(g_2, [-1, s4, s4, 2*f_dim])
        g_2 = tf.layers.batch_normalization(g_2, training=phase_train, epsilon=1e-8)
        # g_2 = batch_norm(g_2)
        g_2 = tf.nn.relu(g_2)
        print ("G2: ", g_2)

        g_3 = tf.concat(axis=3, values=[g_2, tf.tile(yb, [1,s4,s4,1])])
        g_3 = tf.nn.conv2d_transpose(g_3, weights['wg3'], (batch_size, s2, s2, 1*f_dim), strides=strides, padding="SAME")
        # g_3 = tf.nn.bias_add(g_3, biases['bg3'])
        g_3 = tf.layers.batch_normalization(g_3, training=phase_train, epsilon=1e-8)
        # g_3 = batch_norm(g_3)
        g_3 = tf.nn.relu(g_3)
        print ("G3: ", g_3)

        g_4 = tf.concat(axis=3, values=[g_3, tf.tile(yb, [1,s2,s2,1])])
        g_4 = tf.nn.conv2d_transpose(g_4, weights['wg4'], (batch_size, s, s, img_channel), strides=strides, padding="SAME")
        # g_4 = tf.nn.bias_add(g_4, biases['bg4'])
        g_4 = tf.nn.sigmoid(g_4)
        print ("G4: ", g_4)

    return g_4

# --- 
def discriminator(inputs, y, phase_train=True, reuse=False):

    with tf.variable_scope("dis", reuse=reuse):
        yb = tf.reshape(y, tf.stack([-1, 1, 1, y_size]))
        inputs = tf.concat(axis=3, values=[inputs, tf.tile(yb, [1,s,s,1])])

        d_1 = tf.nn.conv2d(inputs, weights['wd1'], strides=strides, padding="SAME")
        # d_1 = tf.nn.bias_add(d_1, biases['bd1'])
        d_1 = lrelu(d_1, leak_value)
        print('D1: ', d_1)

        d_2 = tf.concat(axis=3, values=[d_1, tf.tile(yb, [1,s2,s2,1])])
        d_2 = tf.nn.conv2d(d_2, weights['wd2'], strides=strides, padding="SAME") 
        # d_2 = tf.nn.bias_add(d_2, biases['bd2'])
        d_2 = tf.layers.batch_normalization(d_2, training=phase_train, epsilon=1e-8)
        # d_2 = batch_norm(d_2)
        d_2 = lrelu(d_2, leak_value)
        print('D2: ', d_2)

        shape = d_2.shape.as_list()
        dim = np.prod(shape[1:])
        d_3 = tf.reshape(d_2, [-1, dim])
        d_3 = tf.concat(axis=1, values=[d_3, y])
        d_3 = tf.matmul(d_3, weights['wd3'])
        # d_3 = tf.nn.bias_add(d_3, biases['bd3'])
        d_3 = tf.layers.batch_normalization(d_3, training=phase_train, epsilon=1e-8)
        # d_3 = batch_norm(d_3)
        d_3 = lrelu(d_3, leak_value)
        print('D3: ', d_3)

        d_4 = tf.concat(axis=1, values=[d_3, y])
        d_4 = tf.matmul(d_4, weights['wd4'])
        # d_3_no_sigmoid = tf.add(d_3, biases['bd3'])
        # d_3 = tf.nn.sigmoid(tf.add(d_3, biases['bd3']))
        # d_4_no_sigmoid = tf.clip_by_value(d_4, 1e-7, 1.-1e-7)
        d_4_no_sigmoid = d_4
        d_4 = tf.nn.tanh(d_4)
        print('D4: ', d_4)

    return d_4, d_4_no_sigmoid
# --- ]

# --- [ Initialize Global Variables: strides, sizes, weights, biases
def init():

    global strides, s, s2, s4, s8, s16

    strides    = [1,2,2,1]

    s   = int(img_len)  #64
    s2  = int(np.ceil(float(s) / float(strides[1])))    #32
    s4  = int(np.ceil(float(s2) / float(strides[1])))   #16
    s8  = int(np.ceil(float(s4) / float(strides[1])))   #8
    s16 = int(np.ceil(float(s8) / float(strides[1])))   #4

    initializeWeights()
    # initializeBiases()

# --- 
def initializeWeights():

    global weights
    # weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
    weight_init = tf.random_normal_initializer(stddev=0.02)

    weights['wg1'] = tf.get_variable("wg1", shape=[z_size+y_size, 1024], initializer=weight_init)
    weights['wg2'] = tf.get_variable("wg2", shape=[1024+y_size, s4*s4*2*f_dim], initializer=weight_init)
    weights['wg3'] = tf.get_variable("wg3", shape=[5, 5, 1*f_dim, 2*f_dim+y_size], initializer=weight_init)
    weights['wg4'] = tf.get_variable("wg4", shape=[5, 5, img_channel, 1*f_dim+y_size], initializer=weight_init)

    weights['wd1'] = tf.get_variable("wd1", shape=[5, 5, img_channel+y_size, 1*f_dim], initializer=weight_init)
    weights['wd2'] = tf.get_variable("wd2", shape=[5, 5, 1*f_dim+y_size, 2*f_dim], initializer=weight_init)
    weights['wd3'] = tf.get_variable("wd3", shape=[s4*s4*2*f_dim+y_size, 1024], initializer=weight_init)
    weights['wd4'] = tf.get_variable("wd4", shape=[1024+y_size, 1], initializer=weight_init)

    return weights

# ---
def initializeBiases():
    
    global biases
    zero_init = tf.zeros_initializer()

    biases['bg1'] = tf.get_variable("bg1", shape=[s4*s4*4*f_dim], initializer=zero_init)
    biases['bg2'] = tf.get_variable("bg2", shape=[1*f_dim], initializer=zero_init)
    biases['bg3'] = tf.get_variable("bg3", shape=[img_channel], initializer=zero_init)

    biases['bd1'] = tf.get_variable("bd1", shape=[1*f_dim], initializer=zero_init)
    biases['bd2'] = tf.get_variable("bd2", shape=[4*f_dim], initializer=zero_init)
    biases['bd3'] = tf.get_variable("bd3", shape=[1024], initializer=zero_init) 

    return biases
# --- ]

def trainGAN(is_dummy=False, checkpoint=None, name=model_name):

    init()

    global_step = tf.get_variable('global_step', shape=[], initializer=tf.zeros_initializer(), dtype=tf.int32)

    x_vector = tf.placeholder(shape=[batch_size, img_len, img_len, img_channel],dtype=tf.float32) 
    y_vector = tf.placeholder(shape=[batch_size, y_size],dtype=tf.float32)
    z_vector = tf.placeholder(shape=[batch_size, z_size],dtype=tf.float32) 
    print ('X: ', x_vector)
    print ('Y: ', y_vector)
    print ('Z: ', z_vector)

    net_g_train = generator(z_vector, y_vector, phase_train=True, reuse=False) 
    print ('G_train: ', net_g_train)

    d_output_ax, d_output_x = discriminator(x_vector, y_vector, phase_train=True, reuse=False)
    # d_output_ax = tf.clip_by_value(d_output_ax, 1e-7, 1.-1e-7)
    # summary_d_x_hist = tf.summary.histogram("d_prob_x", d_output_x)

    d_output_az, d_output_z = discriminator(net_g_train, y_vector, phase_train=True, reuse=True)
    # d_output_az = tf.clip_by_value(d_output_az, 1e-7, 1.-1e-7)
    # summary_d_z_hist = tf.summary.histogram("d_prob_z", d_output_z)

    # Compute the discriminator accuracy (probability > 0.5 => acc = 1)
    # `n_p_x` : the number of D(x) output which prob approximate to 1
    # `n_p_z` : the number of D(G(z)) output which prob approximate to 0
    d_output_ax = tf.reshape(d_output_ax, [-1])
    d_output_az = tf.reshape(d_output_az, [-1])
    n_p_x = tf.reduce_sum(tf.cast(d_output_ax > 0, tf.int32))  # hope all d_output_x ~ 1
    n_p_z = tf.reduce_sum(tf.cast(d_output_az <= 0, tf.int32)) # hope all d_output_z ~ 0
    d_acc = tf.divide( n_p_x + n_p_z, 2 * np.prod(d_output_ax.shape.as_list()) )
    # _, n_p_x = tf.nn.moments(tf.nn.softmax(d_output_ax), axes=[0])
    # _, n_p_z = tf.nn.moments(tf.nn.softmax(d_output_az), axes=[0])
    # d_acc = tf.add(n_p_x, n_p_z)

    # Compute the discriminator and generator loss
    # --- [ Bad Accuracy
    # d_loss = -tf.reduce_mean(tf.log(d_output_ax) + tf.log(1-d_output_az))
    # g_loss = -tf.reduce_mean(tf.log(d_output_az))

    # --- [ Sigmoid BCE
    # d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=tf.ones_like(d_output_x)) + \
             # tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=tf.zeros_like(d_output_z))
    # g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=tf.ones_like(d_output_z)) + \
             # tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=tf.zeros_like(d_output_x))

    # --- [ Trick: label smoothing
    # d_output_shape = d_output_x.get_shape().as_list()
    # smooth_ones  = tf.random_uniform(d_output_shape, 0.7, 1.0)
    # smooth_zeros = tf.random_uniform(d_output_shape, 0.0, 0.3)

    # d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=smooth_ones) + \
             # tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=smooth_zeros)
    # g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=smooth_ones) + \
             # tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=smooth_zeros)

    # --- [ Softmax BCE
    logits = tf.concat( (d_output_ax, d_output_az), axis=0 )

    # d_ground_truth = tf.concat( (tf.ones_like(d_output_ax), tf.zeros_like(d_output_az)), axis=0 )
    # g_ground_truth = tf.concat( (tf.zeros_like(d_output_ax), tf.ones_like(d_output_az)), axis=0 )
    d_ground_truth = tf.concat( (tf.random_uniform([batch_size], 0.7, 1.0), -tf.random_uniform([batch_size], 0.7, 1.0)), axis=0 )
    g_ground_truth = tf.concat( (-tf.random_uniform([batch_size], 0.7, 1.0), tf.random_uniform([batch_size], 0.7, 1.0)), axis=0 )

    d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=d_ground_truth)
    g_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=g_ground_truth)

    print ("D_loss: ", d_loss)
    print ("G_loss: ", g_loss)

    # summary_d_loss = tf.summary.scalar("d_loss", d_loss)
    # summary_g_loss = tf.summary.scalar("g_loss", g_loss)
    # summary_n_p_z = tf.summary.scalar("n_p_z", n_p_z)
    # summary_n_p_x = tf.summary.scalar("n_p_x", n_p_x)
    # summary_d_acc = tf.summary.scalar("d_acc", d_acc)

    net_g_test = generator(z_vector, y_vector, phase_train=False, reuse=True)
    print ('G_test: ', net_g_test)

    para_g = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg', 'bg', 'gen'])]
    para_d = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wd', 'bd', 'dis'])]

    d_lr_decay = tf.train.exponential_decay(d_lr, global_step, 100, 0.96, staircase=True)
    g_lr_decay = tf.train.exponential_decay(g_lr, global_step, 100, 0.96, staircase=True)

    optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr_decay,beta1=beta).minimize(d_loss,var_list=para_d)
    optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr_decay,beta1=beta).minimize(g_loss,var_list=para_g)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

    # Save summary
    # if not os.path.exists(summary_dir):
        # os.makedirs(summary_dir)
    # writer = tf.summary.FileWriter(summary_dir)


    ### Constraint GPU memory usage
    # [ref] https://stackoverflow.com/questions/34199233
    # [ref] https://indico.io/blog/the-good-bad-ugly-of-tensorflow/
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:  
      
        sess.run(tf.global_variables_initializer()) 

        if checkpoint is not None:
            saver.restore(sess, checkpoint) 
            epoch = sess.run(global_step)
            sess.run(tf.assign(global_step, epoch + 1))

        if is_dummy:
            train = np.random.randint(0,1,(batch_size,img_len,img_len,img_channel))
            print ('Using Dummy Data')
        else:
            mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
            train  = mnist.train.images.reshape(-1, 28, 28, 1)
            labels = mnist.train.labels.reshape(-1, 10)
            # train = np.zeros((len(orig), img_len, img_len))
            # for i in range(len(orig)):
                # train[i] = scipy.misc.imresize(orig[i], (img_len, img_len))
            # train = train[..., np.newaxis] / 255.
            print ('Using mnist Data')
            print ('[!] train shape: ', train.shape)
            print ('[!] train (min, max): ', train.min(), train.max())
        # --- CENTRAL ---
        # train /= (train.max() - train.min())
        # train = train * 2. - 1.
        # ---------------

        z_val = np.random.normal(0, 0.33, size=[batch_size, z_size]).astype(np.float32)
        # z_val = np.random.uniform(-1, 1, size=[batch_size, z_size]).astype(np.float32)
        y_val = one_hot([batch_size, y_size])

        for epoch in range(sess.run(global_step), n_epochs):
            
            idx = np.random.randint(len(train), size=batch_size)
            x = train[idx]
            y = labels[idx]
            z = np.random.normal(0, 0.33, size=[batch_size, z_size]).astype(np.float32)
            # z = np.random.uniform(-1, 1, size=[batch_size, z_size]).astype(np.float32)

            # Update the discriminator and generator
            # d_summary_merge = tf.summary.merge([summary_d_loss,
                                                # summary_d_x_hist, 
                                                # summary_d_z_hist,
                                                # summary_n_p_x,
                                                # summary_n_p_z,
                                                # summary_d_acc])

            _,  discriminator_loss = sess.run([optimizer_op_d,d_loss],feed_dict={z_vector:z, x_vector:x, y_vector:y})
            _,  generator_loss, D_z = sess.run([optimizer_op_g,g_loss, d_output_z],feed_dict={z_vector:z, y_vector:y, x_vector:x})
            # _, summary_d, discriminator_loss = sess.run([optimizer_op_d,d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
            # _, summary_g, generator_loss = sess.run([optimizer_op_g,summary_g_loss,g_loss],feed_dict={z_vector:z})  
            d_accuracy, n_x, n_z = sess.run([d_acc, n_p_x, n_p_z],feed_dict={z_vector:z, x_vector:x, y_vector:y})
            print ("# of (D(x) -> 1) : ", n_x)
            print ("# of (D(G(z)) -> 0) : ", n_z)
            print ('Discriminator Training ', "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss, "d_acc: ", d_accuracy)
            print ('Generator Training ', "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:', generator_loss, "d_acc: ", d_accuracy)

            # if d_accuracy < d_thresh:
                # _, discriminator_loss = sess.run([optimizer_op_d,d_loss],feed_dict={z_vector:z, x_vector:x})
                # # _, discriminator_loss = sess.run([optimizer_op_d,d_loss],feed_dict={z_vector:z, x_vector:x})
                # generator_loss = sess.run(g_loss,feed_dict={z_vector:z})  
                # # _, summary_d, discriminator_loss = sess.run([optimizer_op_d,d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
                # # _, summary_d, discriminator_loss = sess.run([optimizer_op_d,d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
                # # summary_g, generator_loss = sess.run([summary_g_loss,g_loss],feed_dict={z_vector:z})  
                # d_accuracy = sess.run(d_acc,feed_dict={z_vector:z, x_vector:x})
                # print ('Discriminator Training ', "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss, "d_acc: ", d_accuracy)
            # else:
            #
            ### TODO ###
            # if d_accuracy > d_thresh:
                # _, generator_loss = sess.run([optimizer_op_g,g_loss],feed_dict={z_vector:z, x_vector:x})
                # while generator_loss > 1.5:
                    # _, generator_loss = sess.run([optimizer_op_g,g_loss],feed_dict={z_vector:z, x_vector:x})
                # # summary_d, discriminator_loss = sess.run([d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
                # d_accuracy = sess.run(d_acc,feed_dict={z_vector:z, x_vector:x})
                # print ('Generator Training ', "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss, "d_acc: ", d_accuracy)


            # output generated chairs
            if epoch % 100 == 0:
                g_train = sess.run(net_g_test,feed_dict={z_vector:z, y_vector:y}) #type=np.ndarray
                g_val   = sess.run(net_g_test,feed_dict={z_vector:z_val, y_vector:y_val}) #type=np.ndarray
                if not os.path.exists(train_sample_directory):
                    os.makedirs(train_sample_directory)
                save_visualization(g_train, save_path=os.path.join(train_sample_directory, '{}_{}.jpg'.format(name, epoch)))
                save_visualization(g_val, save_path=os.path.join(train_sample_directory, '{}_val_{}.jpg'.format(name, epoch)))
                save_visualization(x, save_path=os.path.join(train_sample_directory, '{}_real_{}.jpg'.format(name, epoch)))
                # --- [ in Python2, dump as a pickle file. Loading this file by `np.load(filename, encoding='latin1')` in Python3 ] ---
                # g_val.dump(os.path.join(train_sample_directory, '{}_{}.pkl'.format(name, epoch)))
                # ---------------------------------------------------------------------------------------------------------------------

            # store checkpoint
            if epoch % 1000 == 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)      
                saver.save(sess, save_path = os.path.join(model_directory, '{}.ckpt'.format(name)), global_step=global_step)

            # writer.add_summary(summary_d, epoch)
            # writer.add_summary(summary_g, epoch)
            # writer.flush()
            sess.run(tf.assign(global_step, epoch + 1))


def testGAN(trained_model_path=None, n_batches=batch_size):

    init()

    z_vector = tf.placeholder(shape=[batch_size,z_size],dtype=tf.float32) 
    net_g_test = generator(z_vector, phase_train=True, reuse=True)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, trained_model_path) 
        i = 0

        while True:
            stddev = 0.33; i += 1
            try:
                next_sigma = float( input('Please enter the standard deviation of normal distribution [{}]: '.format(stddev)) or stddev )
                z_sample = np.random.normal(0, next_sigma, size=[batch_size, z_size]).astype(np.float32)
                g_objects = sess.run(net_g_test,feed_dict={z_vector:z_sample})
                save_visualization(g_objects, save_path='test_{}_{}.jpg'.format(i, next_sigma))
            except:
                break


def save_visualization(objs, size=(12,12), save_path='./train_sample/sample.jpg'):
    h, w = objs.shape[1], objs.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for n, obj in enumerate(objs):
        row = int(n / size[1])
        col = n % size[1]
        img[row*h:(row+1)*h, col*w:(col+1)*w, :] = obj

    scipy.misc.imsave(save_path, img)
    print ('[!] Save ', save_path)


def main(_):
    if FLAGS.train:
        trainGAN(is_dummy=FLAGS.dummy, checkpoint=FLAGS.ckpath, name=FLAGS.name)
    else:
        if FLAGS.ckpath:
            testGAN(train_model_path=FLAGS.ckpath)
        else:
            logger.error("Needs checkpoint path.")


if __name__ == '__main__':
    tf.app.run()

