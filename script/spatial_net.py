from VGG import VGG16
import tensorflow as tf
import numpy as np

def spatial_aware(input_feature, dimension, trainable, name):
    batch, height, width, channel = input_feature.get_shape().as_list()
    vec1 = tf.reshape(tf.reduce_max(input_feature, axis=-1), [-1, height * width])

    with tf.variable_scope(name):
        weight1 = tf.get_variable(name='weights1', shape=[height * width, int(height * width/2), dimension],
                                 trainable=trainable,
                                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias1 = tf.get_variable(name='biases1', shape=[1, int(height * width/2), dimension],
                               trainable=trainable, initializer=tf.constant_initializer(0.1),
                               regularizer=tf.contrib.layers.l1_regularizer(0.01))
        # vec2 = tf.matmul(vec1, weight1) + bias1
        vec2 = tf.einsum('bi, ijd -> bjd', vec1, weight1) + bias1


        weight2 = tf.get_variable(name='weights2', shape=[int(height * width / 2), height * width, dimension],
                                  trainable=trainable,
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias2 = tf.get_variable(name='biases2', shape=[1, height * width, dimension],
                                trainable=trainable, initializer=tf.constant_initializer(0.1),
                                regularizer=tf.contrib.layers.l1_regularizer(0.01))
        vec3 = tf.einsum('bjd, jid -> bid', vec2, weight2) + bias2

        return vec3


def SAFA(x_sat, x_grd, keep_prob, dimension, trainable):

    vgg_grd = VGG16()
    grd_local = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_local = tf.nn.max_pool(grd_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    batch, g_height, g_width, channel = grd_local.get_shape().as_list()

    grd_w = spatial_aware(grd_local, dimension, trainable, name='spatial_grd')
    grd_local = tf.reshape(grd_local, [-1, g_height * g_width, channel])

    grd_global = tf.einsum('bic, bid -> bdc', grd_local, grd_w)
    grd_global = tf.reshape(grd_global, [-1, dimension*channel])


    vgg_sat = VGG16()
    sat_local = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_local = tf.nn.max_pool(sat_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    batch, s_height, s_width, channel = sat_local.get_shape().as_list()

    sat_w = spatial_aware(sat_local, dimension, trainable, name='spatial_sat')
    sat_local = tf.reshape(sat_local, [-1, s_height * s_width, channel])

    sat_global = tf.einsum('bic, bid -> bdc', sat_local, sat_w)
    sat_global = tf.reshape(sat_global, [-1, dimension*channel])

    return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)

#calculate kernel size and stride
def adaptive_pooling(input, input_h, input_w, output_h, output_w):
    stride_h = np.floor(input_h / output_h).astype(np.int32)
    stride_w = np.floor(input_w / output_w).astype(np.int32)
    kernel_h = input_h - (output_h - 1) * stride_h
    kernel_w = input_w - (output_w - 1) * stride_w
    adapool = tf.nn.avg_pool2d(input, [1, kernel_h, kernel_w, 1], [1, stride_h, stride_w, 1], padding='VALID')
    return adapool

#block spatial aware column partition
def get_block_feature(input_feature, dimension):
    batch, height, width, channel = input_feature.get_shape().as_list()
    sw = np.floor(width / dimension).astype(np.int32) #stride
    kw = width - (dimension - 1) * sw #kernel
    f_bs = []
    for i in range(dimension):
        f_b = input_feature[:,:,i*sw:i*sw+kw,:]
        f_bs.append(f_b)
    block_feature = tf.stack(f_bs,axis=4)   #batch*4*6*512*8
    return block_feature

#block spatial aware row partition
# def get_block_feature(input_feature, dimension):
#     batch, height, width, channel = input_feature.get_shape().as_list()
#     sh = np.floor(height / dimension).astype(np.int32) #stride
#     print(sh)
#     if sh < 1:
#         input_feature = tf.image.resize_bilinear(input_feature, [dimension, width], align_corners=True)
#         batch, height, width, channel = input_feature.get_shape().as_list()
#         sh = np.floor(height / dimension).astype(np.int32)
#         print('new',sh)
#     kh = height - (dimension - 1) * sh #kernel
#     f_bs = []
#     for i in range(dimension):
#         f_b = input_feature[:,i*sh:i*sh+kh,:,:]
#         f_bs.append(f_b)
#     block_feature = tf.stack(f_bs,axis=4)   #batch*1*20*512*8
#     return block_feature

def block_spatial_aware(block_feature, dimension, trainable, name):
    
    batch, height, width, channel, block = block_feature.get_shape().as_list()
    vec1 = tf.reshape(tf.reduce_max(block_feature, axis=-2), [-1, height * width, dimension])
    with tf.variable_scope(name):
        weight1 = tf.get_variable(name='weights1', shape=[height * width, int(height * width/2), dimension],
                                trainable=trainable,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias1 = tf.get_variable(name='biases1', shape=[1, int(height * width/2), dimension],
                            trainable=trainable, initializer=tf.constant_initializer(0.1),
                            regularizer=tf.contrib.layers.l1_regularizer(0.01))
        # vec2 = tf.matmul(vec1, weight1) + bias1
        vec2 = tf.einsum('bid, ijd -> bjd', vec1, weight1) + bias1   #batch*12*8 column partition/batch*10*8:row partition


        weight2 = tf.get_variable(name='weights2', shape=[int(height * width / 2), height * width, dimension],
                                trainable=trainable,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias2 = tf.get_variable(name='biases2', shape=[1, height * width, dimension],
                                trainable=trainable, initializer=tf.constant_initializer(0.1),
                                regularizer=tf.contrib.layers.l1_regularizer(0.01))
        vec3 = tf.einsum('bjd, jid -> bid', vec2, weight2) + bias2  #batch*24*8/batch*20*8

        return vec3

def LPN_AWARE(x_sat, x_grd, keep_prob, dimension, trainable, multi_loss):
    vgg_grd = VGG16()
    grd_local = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_local = tf.nn.max_pool(grd_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')   #b*4*20*512
    grd_local = get_block_feature(grd_local, dimension)  #batch*4*6*512*8
    batch, g_height, g_width, channel, dimension = grd_local.get_shape().as_list()
    # print(grd_local)
    grd_w = block_spatial_aware(grd_local, dimension, trainable, name='b_spatial_grd') #b*24*8
    grd_local = tf.reshape(grd_local, [-1, g_height * g_width, channel, dimension])

    grd_global_ = tf.einsum('bicd, bid -> bdc', grd_local, grd_w) #b*8*512
    grd_global = tf.reshape(grd_global_, [-1, dimension*channel])
    # print('grd_block: ', grd_block.shape)

    vgg_sat = VGG16()
    sat_local = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_local = tf.nn.max_pool(sat_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    sat_local = get_block_feature(sat_local, dimension)  #batch*4*6*512*8
    batch, g_height, g_width, channel, dimension = sat_local.get_shape().as_list()

    sat_w = block_spatial_aware(sat_local, dimension, trainable, name='b_spatial_sat') #b*24*8
    sat_local = tf.reshape(sat_local, [-1, g_height * g_width, channel, dimension])

    sat_global_ = tf.einsum('bicd, bid -> bdc', sat_local, sat_w) #b*8*512
    sat_global = tf.reshape(sat_global_, [-1, dimension*channel])
    # print('sat_block: ', sat_block.shape)
    if multi_loss:
        return tf.nn.l2_normalize(sat_global_, dim=2), tf.nn.l2_normalize(grd_global_, dim=2)
    else:
        return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)
#visual heatmap
def LPN_AWARE_HEATMAP(x_sat, x_grd, keep_prob, dimension, trainable, multi_loss):
    vgg_grd = VGG16()
    grd_local = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_local = tf.nn.max_pool(grd_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')   #b*4*20*512
    grd_local_v = grd_local
    grd_local = get_block_feature(grd_local, dimension)  #batch*4*6*512*8
    batch, g_height, g_width, channel, dimension = grd_local.get_shape().as_list()
    # print(grd_local)
    grd_w = block_spatial_aware(grd_local, dimension, trainable, name='b_spatial_grd') #b*24*8
    grd_local = tf.reshape(grd_local, [-1, g_height * g_width, channel, dimension])

    grd_global_ = tf.einsum('bicd, bid -> bdc', grd_local, grd_w) #b*8*512
    grd_global = tf.reshape(grd_global_, [-1, dimension*channel])
    # print('grd_block: ', grd_block.shape)

    vgg_sat = VGG16()
    sat_local = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_local = tf.nn.max_pool(sat_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    sat_local = get_block_feature(sat_local, dimension)  #batch*4*6*512*8
    batch, g_height, g_width, channel, dimension = sat_local.get_shape().as_list()

    sat_w = block_spatial_aware(sat_local, dimension, trainable, name='b_spatial_sat') #b*24*8
    sat_local = tf.reshape(sat_local, [-1, g_height * g_width, channel, dimension])

    sat_global_ = tf.einsum('bicd, bid -> bdc', sat_local, sat_w) #b*8*512
    sat_global = tf.reshape(sat_global_, [-1, dimension*channel])
    # print('sat_block: ', sat_block.shape)
    if multi_loss:
        return grd_local_v
    else:
        return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)


def VGG_gp(x_sat, x_grd, keep_prob, trainable):

    ############## VGG module #################

    vgg_grd = VGG16()
    grd_vgg = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd')

    vgg_sat = VGG16()
    sat_vgg = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat')

    grd_height, grd_width, grd_channel = grd_vgg.get_shape().as_list()[1:]
    grd_global = tf.nn.max_pool(grd_vgg, [1, grd_height, grd_width, 1], [1, 1, 1, 1], padding='VALID')
    grd_global = tf.reshape(grd_global, [-1, grd_channel])

    sat_height, sat_width, sat_channel = sat_vgg.get_shape().as_list()[1:]
    sat_global = tf.nn.max_pool(sat_vgg, [1, sat_height, sat_width, 1], [1, 1, 1, 1], padding='VALID')
    sat_global = tf.reshape(sat_global, [-1, sat_channel])


    return tf.nn.l2_normalize(sat_global, dim=1), tf.nn.l2_normalize(grd_global, dim=1)


if __name__ == '__main__':
    import numpy as np
    sat_x = np.random.rand(2,112,616,3)
    k=0.8
    x1 = tf.placeholder(tf.float32, [2, 112, 616, 3],name='x1')
    x2 = tf.placeholder(tf.float32, [2, 112, 616, 3],name='x2')
    keep_prob = tf.placeholder(tf.float32)
    dimension = 8
    trainable = True

    out1 = LPN_AWARE(x1, x2, keep_prob, dimension, trainable,True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # output = sess.run(out1, {x1: sat_x})
        output = sess.run(out1, {x1: sat_x, x2: sat_x, keep_prob: k})
        print(output[0].shape)
