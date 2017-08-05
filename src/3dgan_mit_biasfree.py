#!/usr/bin/env python3
"""
Usage Examples:
    [1] python 3dgan_mit_biasfree.py --name biasfree
    [2] python 3dgan_mit_biasfree.py --ckpath models/biasfree.ckpt-100
"""
import os
import sys
import visdom
import logger

import numpy as np
import tensorflow as tf

from tqdm import *
from utils.utils import *
import utils.dataIO as d

'''
Global Parameters
'''
n_epochs   = 10000
batch_size = 32
g_lr       = 0.00025
d_lr       = 0.00001
beta       = 0.5
momentum   = 0.9
d_thresh   = 0.8
z_size     = 200
leak_value = 0.2
obj_ratio  = 0.7
obj        = 'chair' 

cube_len    = 64
cube_channel= 1

f_dim      = 64

model_name = 'biasfree_tfbn'
summary_dir = './summary/'
train_sample_directory = './train_sample/'
model_directory = './models/'
is_local = False

weights, biases = {}, {}

'''
Flags
'''
args = tf.app.flags
args.DEFINE_boolean('train', True, 'True for training, False for testing [%(default)s]')
args.DEFINE_boolean('dummy', False, 'True for random sampling as real input, False for ModelNet voxels as real input [%(default)s]')
args.DEFINE_string('name', model_name, 'Model name [%(default)s]')
args.DEFINE_string('ckpath', None, 'Checkpoint path for restoring (ex: models/biasfree_tfbn.ckpt-100) [%(default)s]')
# args.DEFINE_integer('f_dim', f_dim, 'Dimension of first layer in D [%(default)s]')
FLAGS = args.FLAGS

def generator(z, batch_size=batch_size, phase_train=True, reuse=False):

    strides    = [1,2,2,2,1]

    with tf.variable_scope("gen", reuse=reuse):
        z = tf.reshape(z, (batch_size, 1, 1, 1, z_size))
        g_1 = tf.nn.conv3d_transpose(z, weights['wg1'], (batch_size,4,4,4,512), strides=[1,1,1,1,1], padding="VALID")
        # g_1 = tf.nn.bias_add(g_1, biases['bg1'])                                  
        # g_1 = tf.layers.batch_normalization(g_1, training=phase_train, epsilon=1e-4, momentum=0.9)
        g_1 = tf.nn.relu(g_1)
        print ('G1: ', g_1)

        g_2 = tf.nn.conv3d_transpose(g_1, weights['wg2'], (batch_size,8,8,8,256), strides=strides, padding="SAME")
        # g_2 = tf.nn.bias_add(g_2, biases['bg2'])
        g_2 = tf.layers.batch_normalization(g_2, training=phase_train, epsilon=1e-4, momentum=0.9)
        g_2 = tf.nn.relu(g_2)
        print ('G2: ', g_2)

        g_3 = tf.nn.conv3d_transpose(g_2, weights['wg3'], (batch_size,16,16,16,128), strides=strides, padding="SAME")
        # g_3 = tf.nn.bias_add(g_3, biases['bg3'])
        g_3 = tf.layers.batch_normalization(g_3, training=phase_train, epsilon=1e-4, momentum=0.9)
        g_3 = tf.nn.relu(g_3)
        print ('G3: ', g_3)

        g_4 = tf.nn.conv3d_transpose(g_3, weights['wg4'], (batch_size,32,32,32,64), strides=strides, padding="SAME")
        # g_4 = tf.nn.bias_add(g_4, biases['bg4'])
        g_4 = tf.layers.batch_normalization(g_4, training=phase_train, epsilon=1e-4, momentum=0.9)
        g_4 = tf.nn.relu(g_4)
        print ('G4: ', g_4)
        
        g_5 = tf.nn.conv3d_transpose(g_4, weights['wg5'], (batch_size,64,64,64,1), strides=strides, padding="SAME")
        # g_5 = tf.nn.bias_add(g_5, biases['bg5'])
        # g_5 = tf.nn.sigmoid(g_5)
        g_5 = tf.nn.tanh(g_5)
        print ('G5: ', g_5)
    
    return g_5


def discriminator(inputs, phase_train=True, reuse=False):

    strides    = [1,2,2,2,1]
    with tf.variable_scope("dis", reuse=reuse):
        d_1 = tf.nn.conv3d(inputs, weights['wd1'], strides=strides, padding="SAME")
        # d_1 = tf.nn.bias_add(d_1, biases['bd1'])
        # d_1 = tf.layers.batch_normalization(d_1, training=phase_train, epsilon=1e-4, momentum=0.9)                               
        d_1 = lrelu(d_1, leak_value)
        print ('D1: ', d_1)

        d_2 = tf.nn.conv3d(d_1, weights['wd2'], strides=strides, padding="SAME") 
        # d_2 = tf.nn.bias_add(d_2, biases['bd2'])
        d_2 = tf.layers.batch_normalization(d_2, training=phase_train, epsilon=1e-4, momentum=0.9)
        d_2 = lrelu(d_2, leak_value)
        print ('D2: ', d_2)
        
        d_3 = tf.nn.conv3d(d_2, weights['wd3'], strides=strides, padding="SAME")  
        # d_3 = tf.nn.bias_add(d_3, biases['bd3'])
        d_3 = tf.layers.batch_normalization(d_3, training=phase_train, epsilon=1e-4, momentum=0.9)
        d_3 = lrelu(d_3, leak_value) 
        print ('D3: ', d_3)

        d_4 = tf.nn.conv3d(d_3, weights['wd4'], strides=strides, padding="SAME")     
        # d_4 = tf.nn.bias_add(d_4, biases['bd4'])
        d_4 = tf.layers.batch_normalization(d_4, training=phase_train, epsilon=1e-4, momentum=0.9)
        d_4 = lrelu(d_4)
        print ('D4: ', d_4)


        d_5 = tf.nn.conv3d(d_4, weights['wd5'], strides=[1,1,1,1,1], padding="VALID")     
        d_5_no_sigmoid = d_5
        d_5 = tf.nn.sigmoid(d_5)
        print ('D5: ', d_5)

    return d_5, d_5_no_sigmoid

# --- [ Initialize Global Variables: strides, sizes, weights, biases
def init():

    global strides, s, s2, s4, s8, s16

    strides    = [1,2,2,2,1]

    s   = int(cube_len)  #64
    s2  = int(np.ceil(float(s) / float(strides[1])))    #32
    s4  = int(np.ceil(float(s2) / float(strides[1])))   #16
    s8  = int(np.ceil(float(s4) / float(strides[1])))   #8
    s16 = int(np.ceil(float(s8) / float(strides[1])))   #4

    initializeWeights()
    # initializeBiases()

# --- 
def initializeWeights():

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

# ---
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
# --- ]

def trainGAN(is_dummy=False, checkpoint=None, name=model_name):

    init()

    global_step = tf.get_variable('global_step', shape=[], initializer=tf.zeros_initializer(), dtype=tf.int32)

    z_vector = tf.placeholder(shape=[batch_size,z_size],dtype=tf.float32) 
    x_vector = tf.placeholder(shape=[batch_size,cube_len,cube_len,cube_len,cube_channel],dtype=tf.float32) 

    net_g_train = generator(z_vector, phase_train=True, reuse=False) 
    net_g_test  = generator(z_vector, phase_train=False, reuse=True)
    print ('G train: ', net_g_train)
    print ('G test: ', net_g_test)

    d_output_ax, d_output_x = discriminator(x_vector, phase_train=True, reuse=False)
    d_output_ax = tf.clip_by_value(d_output_ax, 0.01, 0.99)
    # summary_d_x_hist = tf.summary.histogram("d_prob_x", d_output_ax)

    d_output_az, d_output_z = discriminator(net_g_train, phase_train=True, reuse=True)
    d_output_az = tf.clip_by_value(d_output_az, 0.01, 0.99)
    # summary_d_z_hist = tf.summary.histogram("d_prob_z", d_output_az)

    print ('D_output_x: ', d_output_x)
    print ('D_output_z: ', d_output_z)

    # Compute the discriminator accuracy (probability > 0.5 => acc = 1)
    # `n_p_x` : the number of D(x) output which prob > 0.5
    # `n_p_z` : the number of D(G(z)) output which prob <= 0.5
    n_p_x = tf.reduce_sum(tf.cast(d_output_ax > 0.5, tf.int32))  # hope all d_output_ax ~ 1
    n_p_z = tf.reduce_sum(tf.cast(d_output_az <= 0.5, tf.int32)) # hope all d_output_az ~ 0
    d_acc = tf.divide( n_p_x + n_p_z, 2 * np.prod(d_output_ax.shape.as_list()) )

    # Compute the discriminator and generator loss
    # --- [ Bad Accuracy
    # d_loss = -tf.reduce_mean(tf.log(d_output_ax) + tf.log(1-d_output_az))
    # g_loss = -tf.reduce_mean(tf.log(d_output_az))

    d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=tf.ones_like(d_output_x)) + \
             tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=tf.zeros_like(d_output_z))
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=tf.ones_like(d_output_z))
             # tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=tf.zeros_like(d_output_x))
    d_loss = tf.reduce_mean(d_loss)
    g_loss = tf.reduce_mean(g_loss)

    # --- [ Trick: label smoothing
    # d_output_shape = d_output_ax.get_shape().as_list()
    # smooth_ones  = tf.random_uniform(d_output_shape, 0.7, 1.0)
    # smooth_zeros = tf.random_uniform(d_output_shape, 0.0, 0.3)

    # d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=smooth_ones) + \
             # tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=smooth_zeros)
    # g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_z, labels=smooth_ones) + \
             # tf.nn.sigmoid_cross_entropy_with_logits(logits=d_output_x, labels=smooth_zeros)
    # d_loss = tf.reduce_mean(d_loss)
    # g_loss = tf.reduce_mean(g_loss)
    
    # --- [ Softmax BCE (D output: tanh)
    # logits = tf.concat( (d_output_ax, d_output_az), axis=0 )
    # d_ground_truth = tf.concat( (tf.random_uniform([batch_size], 0.7, 1.0), -tf.random_uniform([batch_size], 0.7, 1.0)), axis=0 )
    # g_ground_truth = tf.concat( (-tf.random_uniform([batch_size], 0.7, 1.0), tf.random_uniform([batch_size], 0.7, 1.0)), axis=0 )

    # d_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=d_ground_truth)
    # g_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=g_ground_truth)

    # summary_d_loss = tf.summary.scalar("d_loss", d_loss)
    # summary_g_loss = tf.summary.scalar("g_loss", g_loss)
    # summary_n_p_z = tf.summary.scalar("n_p_z", n_p_z)
    # summary_n_p_x = tf.summary.scalar("n_p_x", n_p_x)
    # summary_d_acc = tf.summary.scalar("d_acc", d_acc)

    net_g_test = generator(z_vector, phase_train=False, reuse=True)

    para_g = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg', 'bg', 'gen'])]
    para_d = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wd', 'bd', 'dis'])]

    # For ``tf.layers.batch_normalization`` updating
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # d_lr_decay = tf.train.exponential_decay(d_lr, global_step, 100, 0.96, staircase=True)
        # g_lr_decay = tf.train.exponential_decay(g_lr, global_step, 100, 0.96, staircase=True)
        optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr,beta1=beta).minimize(d_loss,var_list=para_d)
        optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr,beta1=beta).minimize(g_loss,var_list=para_g)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    # vis = visdom.Visdom()

    # save summary
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
            volumes = np.random.randint(0,2,(batch_size,cube_len,cube_len,cube_len))
            print ('Using Dummy Data')
        else:
            volumes = d.getAll(obj=obj, train=True, is_local=is_local, obj_ratio=obj_ratio)
            print ('Using ' + obj + ' Data')
        volumes = volumes[...,np.newaxis].astype(np.float32)
        # Same as `volumes[:,:,:,:,np.newaxis].astype(np.float32)`
        # --- CENTRAL ---
        # volumes *= 2.0
        # volumes -= 1.0
        # ---------------

        z_val = np.random.normal(0, 0.33, size=[batch_size, z_size]).astype(np.float32)

        for epoch in range(sess.run(global_step), n_epochs):
            
            idx = np.random.randint(len(volumes), size=batch_size)
            x = volumes[idx]
            z = np.random.normal(0, 0.33, size=[batch_size, z_size]).astype(np.float32)
            # z = np.random.uniform(0, 1, size=[batch_size, z_size]).astype(np.float32)

            # Update the discriminator and generator
            # d_summary_merge = tf.summary.merge([summary_d_loss,
                                                # summary_d_x_hist, 
                                                # summary_d_z_hist,
                                                # summary_n_p_x,
                                                # summary_n_p_z,
                                                # summary_d_acc])

            _,  discriminator_loss = sess.run([optimizer_op_d,d_loss],feed_dict={z_vector:z, x_vector:x})
            _,  generator_loss = sess.run([optimizer_op_g,g_loss],feed_dict={z_vector:z, x_vector:x})
            # _, summary_d, discriminator_loss = sess.run([optimizer_op_d,d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
            # _, summary_g, generator_loss = sess.run([optimizer_op_g,summary_g_loss,g_loss],feed_dict={z_vector:z})  
            d_accuracy, n_x, n_z = sess.run([d_acc, n_p_x, n_p_z],feed_dict={z_vector:z, x_vector:x})
            print ("# of (D(x) > 0.5) : ", n_x)
            print ("# of (D(G(z)) <= 0.5) : ", n_z)
            print ('Discriminator Training ', "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss, "d_acc: ", d_accuracy)
            print ('Generator Training ', "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss, "d_acc: ", d_accuracy)

            if d_accuracy < d_thresh:
                _, discriminator_loss = sess.run([optimizer_op_d,d_loss],feed_dict={z_vector:z, x_vector:x})
                d_accuracy = sess.run(d_acc,feed_dict={z_vector:z, x_vector:x})
                print ('Discriminator Training ', "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss, "d_acc: ", d_accuracy)

            # output generated chairs
            if epoch % 200 == 0:
                g_objects = sess.run(net_g_test,feed_dict={z_vector:z_val}) #type=np.ndarray
                if not os.path.exists(train_sample_directory):
                    os.makedirs(train_sample_directory)
                # --- [ in Python2, dump as a pickle file. Loading this file by `np.load(filename, encoding='latin1')` in Python3 ] ---
                g_objects.dump(os.path.join(train_sample_directory, '{}_{}.pkl'.format(model_name, epoch)))
                # ---------------------------------------------------------------------------------------------------------------------
                # id_ch = np.random.randint(0, batch_size, 4)
                # for i in range(4):
                    # if g_objects[id_ch[i]].max() > 0.5:
                        # d.plotMeshFromVoxels(np.squeeze(g_objects[id_ch[i]]>0.5), threshold=0.5)

            # store checkpoint
            if epoch % 1000 == 0:
                if not os.path.exists(model_directory):
                    os.makedirs(model_directory)      
                saver.save(sess, save_path = os.path.join(model_directory, '{}.ckpt'.format(model_name)), global_step=global_step)

            # writer.add_summary(summary_d, epoch)
            # writer.add_summary(summary_g, epoch)
            # writer.flush()
            sess.run(tf.assign(global_step, epoch + 1))


def testGAN(trained_model_path=None, n_batches=40):

    init()

    z_vector = tf.placeholder(shape=[batch_size,z_size],dtype=tf.float32) 
    net_g_test = generator(z_vector, phase_train=True, reuse=True)

    vis = visdom.Visdom()

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, trained_model_path) 
        i = 0
        stddev = 0.33

        while True:
            i += 1
            try:
                next_sigma = float( input('Please enter the standard deviation of normal distribution [{}]: '.format(stddev)) or stddev )
                z_sample = np.random.normal(0, next_sigma, size=[batch_size, z_size]).astype(np.float32)
                g_objects = sess.run(net_g_test,feed_dict={z_vector:z_sample})
                # save_visualization(g_objects, save_path='test_{}_{}.jpg'.format(i, next_sigma))
                scipy.misc.imssave('{}_test_{}_{}.jpg'.format(name, i, next_sigma), g_objects)
                # id_ch = np.random.randint(0, batch_size, 4)
                # for i in range(4):
                    # print (g_objects[id_ch[i]].max(), g_objects[id_ch[i]].min(), g_objects[id_ch[i]].shape)
                    # if g_objects[id_ch[i]].max() > 0.5:
                        # d.plotVoxelVisdom(np.squeeze(g_objects[id_ch[i]]>0.5), vis, '_'.join(map(str,[i])))
            except:
                break

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

