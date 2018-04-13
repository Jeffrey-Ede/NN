from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

slim = tf.contrib.slim #For depthwise separable strided atrous convolutions

tf.logging.set_verbosity(tf.logging.DEBUG)

features1 = 1 #Number of features to use for initial convolution
features2 = 2*features1 #Number of features after 2nd convolution
features3 = 3*features2 #Number of features after 3rd convolution
features4 = 4*features3 #Number of features after 4th convolution
aspp_filters = features4 #Number of features for atrous convolutional spatial pyramid pooling

aspp_rateSmall = 6
aspp_rateMedium = 12
aspp_rateLarge = 18

trainDir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/2100plus_dm3/"
valDir = ""
testDir = ""

model_dir = "E:/stills/"

shuffle_buffer_size = 5000
parallel_readers = 6
prefetch_buffer_size = 1

batch_size = 4 #Batch size to use during training
num_epochs = 1

logDir = "C:/dump/train/"
log_every = 1 #Log every _ examples



#Dimensions of images in the dataset
height = width = 2048
channels = 1 #Greyscale input image

## Initial idea: aspp, batch norm + Leaky PRELU, residual connection and lower feature numbers
def architecture(features, mode):
    """Atrous convolutional encoder-decoder noise-removing network"""

    phase = mode == tf.estimator.ModeKeys.TRAIN
    concat_axis = 3

    ##Reusable blocks

    def conv_block(input, filters, phase=phase):
        """
        Convolution -> batch normalisation -> leaky relu
        phase defaults to true, meaning that the network is being trained
        """

        conv_block = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=3,
            padding="same")

        conv_block = tf.contrib.layers.batch_norm(
            conv_block, 
            center=True, scale=True, 
            is_training=phase)

        conv_block = tf.nn.leaky_relu(
            features=conv_block,
            alpha=0.2)

        return conv_block

    def aspp_block(input, phase=phase):
        """
        Atrous spatial pyramid pooling
        phase defaults to true, meaning that the network is being trained
        """

        #Convolutions at multiple rates
        conv1x1 = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=1,
            padding="same",
            name="1x1")
        conv1x1 = tf.contrib.layers.batch_norm(
            conv1x1, 
            center=True, scale=True, 
            is_training=phase)

        conv3x3_rateSmall = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateSmall,
            name="lowRate")
        conv3x3_rateSmall = tf.contrib.layers.batch_norm(
            conv3x3_rateSmall, 
            center=True, scale=True, 
            is_training=phase)

        conv3x3_rateMedium = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateMedium,
            name="mediumRate")
        conv3x3_rateMedium = tf.contrib.layers.batch_norm(
            conv3x3_rateMedium, 
            center=True, scale=True, 
            is_training=phase)

        conv3x3_rateLarge = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateLarge,
            name="highRate")
        conv3x3_rateLarge = tf.contrib.layers.batch_norm(
            conv3x3_rateLarge, 
            center=True, scale=True, 
            is_training=phase)

        #Image-level features
        pooling = tf.nn.pool(
            input=input,
            window_shape=(2,2),
            pooling_type="AVG",
            padding="SAME")
        #Use 1x1 convolutions to project into a feature space the same size as the atrous convolutions'
        pooling = tf.layers.conv2d(
            inputs=pooling,
            filters=aspp_filters,
            kernel_size=1,
            padding="SAME",
            name="imageLevel")
        pooling = tf.contrib.layers.batch_norm(
            pooling,
            center=True, scale=True,
            is_training=phase)

        #Concatenate the atrous and image-level pooling features
        concatenation = tf.concat(
            values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
            axis=concat_axis)

        #Reduce the number of channels
        reduced = tf.layers.conv2d( #Not sure if this is the correct way to reshape...
            inputs=concatenation,
            filters=aspp_filters,
            kernel_size=1,
            padding="SAME")

        return reduced

    def split_separable_conv2d(
        inputs,
        filters,
        rate=1,
        stride=1,
        weight_decay=0.00004,
        depthwise_weights_initializer_stddev=0.33,
        pointwise_weights_initializer_stddev=0.06,
        scope=''):

        """
        Splits a separable conv2d into depthwise and pointwise conv2d.
        This operation differs from `tf.layers.separable_conv2d` as this operation
        applies activation function between depthwise and pointwise conv2d.
        Args:
        inputs: Input tensor with shape [batch, height, width, channels].
        filters: Number of filters in the 1x1 pointwise convolution.
        rate: Atrous convolution rate for the depthwise convolution.
        weight_decay: The weight decay to use for regularizing the model.
        depthwise_weights_initializer_stddev: The standard deviation of the
            truncated normal weight initializer for depthwise convolution.
        pointwise_weights_initializer_stddev: The standard deviation of the
            truncated normal weight initializer for pointwise convolution.
        scope: Optional scope for the operation.
        Returns:
        Computed features after split separable conv2d.
        """

        outputs = slim.separable_conv2d(
            inputs,
            None,
            3,
            stride=stride,
            depth_multiplier=1,
            rate=rate,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=depthwise_weights_initializer_stddev),
            weights_regularizer=None)

        outputs = tf.contrib.layers.batch_norm(
            outputs, 
            center=True, scale=True, 
            is_training=phase)

        outputs = tf.nn.leaky_relu(
            features=outputs,
            alpha=0.2)

        outputs = slim.conv2d(
            outputs,
            filters,
            kernel_size=1,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=pointwise_weights_initializer_stddev),
            weights_regularizer=slim.l2_regularizer(weight_decay))

        return outputs

    def deconv_block(input, filters, phase=phase):
        '''Transpositionally convolute a feature space to upsample it'''
        
        deconv_block = tf.layers.conv2d_transpose(
            inputs=input,
            filters=filters,
            kernel_size=3,
            strides=2,
            padding="SAME")

        deconv_block = tf.contrib.layers.batch_norm(
            deconv_block, 
            center=True, scale=True, 
            is_training=phase)

        deconv_block = tf.nn.leaky_relu(
            features=deconv_block,
            alpha=0.2)

        return deconv_block

    '''Model building'''
    input_layer = tf.reshape(features, [-1, height, width, channels])

    #Encoding block 1
    cnn1_last = conv_block(
        input=input_layer, 
        filters=features1)
    cnn1_strided = split_separable_conv2d(
        inputs=cnn1_last,
        filters=features1,
        rate=2,
        stride=2)

    #Encoding block 2
    cnn2_last = conv_block(
        input=cnn1_strided,
        filters=features2)
    cnn2_strided = split_separable_conv2d(
        inputs=cnn2_last,
        filters=features2,
        rate=2,
        stride=2)

    #Encoding block 3
    cnn3 = conv_block(
        input=cnn2_strided,
        filters=features3)
    cnn3_last = conv_block(
        input=cnn3,
        filters=features3)
    cnn3_strided = split_separable_conv2d(
        inputs=cnn3_last,
        filters=features3,
        rate=2,
        stride=2)

    #Encoding block 4
    cnn4 = conv_block(
        input=cnn3_strided,
        filters=features4)
    cnn4_last = conv_block(
        input=cnn4,
        filters=features4)
    #cnn4_strided = split_separable_conv2d(
    #    inputs=cnn4_last,
    #    filters=features4,
    #    rate=2,
    #    stride=2)

    ##Atrous spatial pyramid pooling
    #aspp = aspp_block(cnn4_strided)

    aspp = aspp_block(cnn4_last)

    #Upsample the semantics by a factor of 4
    #upsampled_aspp = tf.image.resize_bilinear(
    #    images=aspp,
    #    tf.shape(aspp)[1:3],
    #    align_corners=True)

    #Decoding block 1 (deepest)
    deconv4 = conv_block(aspp, features4)
    deconv4 = conv_block(deconv4, features4)
    
    #Decoding block 2
    deconv4to3 = deconv_block(deconv4, features4)
    concat3 = tf.concat(
        values=[deconv4to3, cnn3_last],
        axis=concat_axis)
    deconv3 = conv_block(concat3, features3)
    deconv3 = conv_block(deconv3, features3)

    #Decoding block 3
    deconv3to2 = deconv_block(deconv3, features3)
    concat2 = tf.concat(
        values=[deconv3to2, cnn2_last],
        axis=concat_axis)
    deconv2 = conv_block(concat2, features2)
    
    #Decoding block 4
    deconv2to1 = deconv_block(deconv2, features2)
    concat1 = tf.concat(
        values=[deconv2to1, cnn1_last],
        axis=concat_axis)
    deconv1 = conv_block(concat1, features1)

    #Create final image with 1x1 convolutions
    deconv_final = tf.layers.conv2d_transpose(
        inputs=deconv1,
        filters=1,
        kernel_size=3,
        padding="SAME")

    #Residually connect the input to the output
    output = input_layer + deconv_final

    #Image values will be between 0 and 1
    output = tf.clip_by_value(
        output,
        clip_value_min=0,
        clip_value_max=1)

    loss = tf.reduce_mean(tf.squared_difference(input_layer, output))

    tf.summary.histogram("loss", loss)

    return loss, output

def load_image(addr, resizeSize=None, imgType=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""
    
    img = imread(addr, mode='F')
    if resizeSize:
        img = cv2.resize(img, resizeSize, interpolation=cv2.INTER_CUBIC)
    img = img.astype(imgType)

    return img

def gen_lq(img, mean):
    '''Generate low quality image'''

    #Ensure that the seed is random
    np.random.seed(int(np.random.rand()*(2**32-1)))

    #Adjust the image scale so that the image has the correct average counts
    lq = np.random.poisson(mean * (img / np.mean(img)))
    
    #Rescale between 0 and 1
    min = np.min(lq)
    lq = (lq-min) / (np.max(lq)-min)

    return lq.astype(np.float32)

def preprocess(lq, img):
    
    return lq, img

def parser(record, mean):
    """Parse files and generate lower quality images from them"""

    img = load_image(record)
    lq = gen_lq(img, mean)

    return preprocess(lq, img)

def input_fn(dir, mean):
    """Create a dataset from a list of filenames"""

    dataset = tf.data.Dataset.list_files(dir+"*.tif")
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(
        lambda file: tuple(tf.py_func(parser, [file, mean], [tf.float32, tf.float32])),
        num_parallel_calls=FLAGS.num_parallel_calls)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    dataset = dataset.repeat(num_epochs)
    
    iter = dataset.make_one_shot_iterator()
    #with tf.Session() as sess:
    #    next = iter.get_next()
    #    print(sess.run(next)) #Output
    #    img = sess.run(next)
        
    lq, img = iter.get_next()

    return lq, img

def main(unused_argv=None):

    mean = 64 #Average value of pixels in low quality generations

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
    with tf.control_dependencies(update_ops):

        lq_train, img_train = input_fn(trainDir, mean)

        loss, prediction = architecture(lq_train, tf.estimator.ModeKeys.TRAIN)
        train_op = tf.train.AdamOptimizer().minimize(loss)

        with tf.Session() as sess: #Alternative is tf.train.MonitoredTrainingSession()

            sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter( logDir, sess.graph )

            counter = 0
            for _ in range(200):
                counter += 1

                merge = tf.summary.merge_all()

                summary, _, loss_value = sess.run([merge, train_op, loss])
                print("Iter: {}, Loss: {:.4f}".format(counter, loss_value))

                train_writer.add_summary(summary, counter)

                #train_writer.add_summary(summary, counter)
                
                #img = sess.run(lq_train)

                #print(np.max(img))

                #cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
                #cv2.imshow("dfsd", img.reshape((2048,2048)))
                #cv2.waitKey(0)

                #sess.run(training_op)


    return 

if __name__ == "__main__":
    tf.app.run()
