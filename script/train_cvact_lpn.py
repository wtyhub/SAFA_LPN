import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from spatial_net import *
from OriNet_CVACT.input_data_act_polar import InputData
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.python.ops.gen_math_ops import *

parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type',              type=str,   help='network type',      default='multi_loss_LPN_AWARE_8')
parser.add_argument('--polar',                     type=int,   help='polar',             default=1)
parser.add_argument('--start_epoch',               type=int,   help='train from epoch',  default=0)
parser.add_argument('--multi_loss',                action='store_true',   help='use multiple loss')
args = parser.parse_args()

# --------------  configuration parameters  -------------- #
# the type of network to be used: "CVM-NET-I" or "CVM-NET-II"
network_type = args.network_type
print('network_type:----------------------', network_type)
polar = args.polar
start_epoch = args.start_epoch
multi_loss = args.multi_loss
data_type = 'CVACT'

batch_size = 32
is_training = True
loss_weight = 10.0
number_of_epoch = 100

learning_rate_val = 1e-5
keep_prob_val = 0.8
# -------------------------------------------------------- #
print('learning rate:-----------------------', learning_rate_val)


def validate(grd_descriptor, sat_descriptor):
    accuracy = 0.0
    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < 1:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy


def compute_loss(sat_global, grd_global, utms_x, UTMthres, batch_hard_count=0):

    with tf.name_scope('weighted_soft_margin_triplet_loss'):

        dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)
        pos_dist = tf.diag_part(dist_array)

        usefulPair = greater_equal(utms_x[:,:,0], UTMthres, 'good_pair')

        usefulPair = tf.cast(usefulPair, tf.float32)

        if batch_hard_count == 0:

            pair_n = tf.reduce_sum(usefulPair)

            # ground to satellite
            triplet_dist_g2s = (pos_dist - dist_array)*usefulPair

            loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

            # satellite to ground
            triplet_dist_s2g = (tf.expand_dims(pos_dist, 1) - dist_array)*usefulPair

            loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

            loss = (loss_g2s + loss_s2g) / 2.0

    return loss

def compute_multi_loss(sat_global, grd_global, utms_x, UTMthres):
    batch, d, channel = sat_global.get_shape().as_list()
    loss = compute_loss(sat_global[:,0,:], grd_global[:,0,:],utms_x, UTMthres)
    for i in range(d-1):
        loss += compute_loss(sat_global[:,i+1,:], grd_global[:,i+1,:], utms_x, UTMthres)
    loss = loss / d
    return loss

def train(start_epoch=0):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 1.
    '''

    # import data
    input_data = InputData(polar)

    # define placeholders
    if polar:
        sat_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name='sat_x')
    else:
        sat_x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='sat_x')

    grd_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name='grd_x')

    utms_x = tf.placeholder(tf.float32, [None, None, 1], name='utms')

    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)

    # build model
    dimension = int(network_type[-1])
    if 'LPN_AWARE' in network_type:
        print('-----------------lpn aware net--------------------------')
        sat_global, grd_global = LPN_AWARE(sat_x, grd_x, keep_prob, dimension, is_training, multi_loss)
    else:# SAFA
        sat_global, grd_global = SAFA(sat_x, grd_x, keep_prob, dimension, is_training)
    # sat_global, grd_global = SAFA(sat_x, grd_x, keep_prob, dimension, is_training)

    if multi_loss:
        batch, d, out_channel = sat_global.get_shape().as_list()
        sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), d*out_channel])
        grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), d*out_channel])
        loss = compute_multi_loss(sat_global, grd_global, utms_x, input_data.posDistSqThr)
    else:
        out_channel = sat_global.get_shape().as_list()[-1]
        sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])
        grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), out_channel])

        loss = compute_loss(sat_global, grd_global, utms_x, input_data.posDistSqThr)


    # set training
    global_step = tf.Variable(0, trainable=False)
    with tf.device('/gpu:0'):
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    global_vars = tf.global_variables()

    var_list = []
    for var in global_vars:
        if 'VGG' in var.op.name and 'Adam' not in var.op.name:
            var_list.append(var)

    saver_to_restore = tf.train.Saver(var_list)

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print('load model...')

        if start_epoch == 0:
            if 'LPN_AWARE' in network_type:
                restore_var = [v for v in tf.global_variables()]
                variables_to_resotre = [v for v in restore_var if v.name.split('/')[0] in ['VGG_grd','VGG_sat']]
                saver_vgg = tf.train.Saver(var_list=variables_to_resotre)
                for i in range(len(variables_to_resotre)):
                    print(variables_to_resotre[i])
                load_model_path = '~/SAFA_LPN/Model/Initialize/initial_model.ckpt'
                saver_vgg.restore(sess, load_model_path)
            else:        
                load_model_path = '~/SAFA_LPN/Model/Initialize/initial_model.ckpt'
                saver.restore(sess, load_model_path)
        else:

            load_model_path = '~/SAFA_LPN/Model/' + data_type + '/' + network_type \
                          + '/polar_' + str(polar) + '/' + str(start_epoch - 1) + '/model.ckpt'
            saver.restore(sess, load_model_path)

        # if start_epoch == 0:
        #     load_model_path = '../Model/Initialize/initial_model.ckpt'
        # else:
        #     load_model_path = './Model/' + data_type + '/' + network_type \
        #                   + '/polar_' + str(polar) + '/' + str(start_epoch - 1) + '/model.ckpt'

        # saver.restore(sess, load_model_path)

        print("   Model loaded from: %s" % load_model_path)
        print('load model...FINISHED')

        # Train
        for epoch in range(start_epoch, number_of_epoch):
            iter = 0
            while True:
                # train
                batch_sat, batch_grd, batch_utm = input_data.next_pair_batch(batch_size)
                if batch_sat is None:
                    break

                global_step_val = tf.train.global_step(sess, global_step)

                feed_dict = {sat_x: batch_sat, grd_x: batch_grd,
                             learning_rate: learning_rate_val, keep_prob: keep_prob_val, utms_x:batch_utm}
                if iter % 20 == 0:
                    _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
                    print('global %d, epoch %d, iter %d: loss : %.4f ' % (global_step_val, epoch, iter, loss_val))

                else:
                    sess.run(train_step, feed_dict=feed_dict)

                iter += 1

            model_dir = '~/SAFA_LPN/Model/' + data_type + '/' + network_type \
                        + '/polar_' + str(polar) + '/' + str(epoch) + '/'

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_path = saver.save(sess, model_dir + 'model.ckpt')
            print("Model saved in file: %s" % save_path)

            # ---------------------- validation ----------------------
            print('validate...')
            print('   compute global descriptors')
            input_data.reset_scan()

            val_i = 0
            while True:
                # print('      progress %d' % val_i)
                batch_sat, batch_grd, _ = input_data.next_batch_scan(batch_size)
                if batch_sat is None:
                    break
                feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
                sat_global_val, grd_global_val = \
                    sess.run([sat_global, grd_global], feed_dict=feed_dict)
                if multi_loss:              #out shape:batch*d*channel
                    batch, d, channel = sat_global_val.shape
                    sat_global_val = sat_global_val.reshape(-1, d*channel)
                    grd_global_val = grd_global_val.reshape(-1, d*channel)
                sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
                grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
                val_i += sat_global_val.shape[0]

            print('   compute accuracy')
            val_accuracy = validate(grd_global_descriptor, sat_global_descriptor)
            print('   %d: accuracy = %.1f' % (epoch, val_accuracy * 100.0))
            file_dir = '~/SAFA_LPN/Result/' + data_type + '/'
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            file = '~/SAFA_LPN/Result/' + data_type + '/' + str(network_type) +'_polar_' + str(polar) + '_accuracy.txt'
            with open(file, 'a') as file:
                file.write(str(epoch) + ' ' + str(iter) + ' : ' + str(val_accuracy) + '\n')



if __name__ == '__main__':
    train(start_epoch)
