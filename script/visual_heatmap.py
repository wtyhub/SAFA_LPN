import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import cv2
import matplotlib.pyplot as plt
from spatial_net import LPN_AWARE_HEATMAP
from input_data_cvusa import InputData
import tensorflow as tf
import numpy as np
import argparse
import scipy.io
parser = argparse.ArgumentParser(description='TensorFlow implementation.')

parser.add_argument('--network_type',              type=str,   help='network type',      default='multi_loss_LPN_AWARE_8')
parser.add_argument('--polar',                     type=int,   help='polar',             default=1)
parser.add_argument('--multi_loss',                action='store_true',   help='use multiple loss')
args = parser.parse_args()

network_type = args.network_type
multi_loss = args.multi_loss
polar = args.polar

data_type = 'CVUSA'

batch_size = 32
is_training = True
loss_weight = 10.0
number_of_epoch = 100

learning_rate_val = 1e-5
keep_prob_val = 0.8

img_root = '../Data/CVUSA/'
grd_path = img_root+'streetview/panos/0027718.jpg'
sat_path = img_root+'polarmap/19/0027718.png'
img_grd = cv2.imread(grd_path)
img_grd = cv2.resize(img_grd, (616, 112), interpolation=cv2.INTER_AREA)
img_grd_dup = img_grd.copy()
img_grd_dup = img_grd_dup.astype(np.float32)
# img -= 100.0
img_grd_dup[:, :, 0] -= 103.939  # Blue
img_grd_dup[:, :, 1] -= 116.779  # Green
img_grd_dup[:, :, 2] -= 123.6  # Red
batch_grd = np.zeros([1, 112, 616, 3], dtype=np.float32)
batch_grd[0,:,:,:] = img_grd_dup

img_sat = cv2.imread(sat_path)
img_sat_dup = img_sat.copy()
img_sat_dup = img_sat_dup.astype(np.float32)
# img -= 100.0
img_sat_dup[:, :, 0] -= 103.939  # Blue
img_sat_dup[:, :, 1] -= 116.779  # Green
img_sat_dup[:, :, 2] -= 123.6  # Red
batch_sat = np.zeros([1, 112, 616, 3], dtype=np.float32)
batch_sat[0,:,:,:] = img_grd_dup

if __name__ == '__main__':
    tf.reset_default_graph()
    sat_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name='sat_x')
    grd_x = tf.placeholder(tf.float32, [None, 112, 616, 3], name='grd_x')
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    dimension = int(network_type[-1])
    grd_feature = LPN_AWARE_HEATMAP(sat_x, grd_x, keep_prob, dimension, is_training, multi_loss)
    batch,H,W,C = grd_feature.get_shape().as_list()
    # heatmap_fea = np.zeros([batch,H,W,C])
    print('setting saver...')
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    print('setting saver done...')

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    print('open session ...')
    with tf.Session(config=config) as sess:
        print('initialize...')
        sess.run(tf.global_variables_initializer())
        
        print('load model...')

        load_model_path = '~/SAFA_LPN/Model/CVUSA/multi_loss_PCB_AWARE_8/polar_1/99/model.ckpt'
        saver.restore(sess, load_model_path)
        feed_dict = {sat_x: batch_sat, grd_x: batch_grd, keep_prob: 1.0}
        heatmap_fea = sess.run(grd_feature, feed_dict=feed_dict)
        print(heatmap_fea.shape)
        heatmap = heatmap_fea.squeeze().sum(2)
        print(heatmap.shape)
        heatmap = cv2.resize(heatmap,(616,112))
        fig = plt.figure()
        heatmap = plt.imshow(heatmap, cmap='viridis')
        plt.axis('off')
        # fig.colorbar(heatmap)
        #plt.show()
        plt.savefig('heatmap_pcb')