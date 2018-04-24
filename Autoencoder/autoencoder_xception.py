from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

import time

import os, random

import warnings

slim = tf.contrib.slim #For depthwise separable strided atrous convolutions

tf.logging.set_verbosity(tf.logging.DEBUG)

filters00 = 32
filters01 = 64
filters1 = 128
filters2 = 256
filters3 = 512
filters4 = 728
filters5 = 1024
filters6 = 1536
numMiddleXception = 16

decode_size1 = 8
decode_channels1 = 256
decode_size2 = 16
decode_channels2 = 192
decode_size3 = 32
decode_channels3 = 128
decode_size4 = 64
decode_channels4 = 96
decode_size5 = 128
decode_channels5 = 64
decode_size6 = 256
decode_channels6 = 48
decode_size7 = 512
decode_channels7 = 32
decode_size8 = 1024
decode_channels8 = 16

fc_features1 = 2048

trainDir = "F:/stills_all/train/"
valDir = "F:/stills_all/val/"
testDir = "F:/stills_all/test/"

modelSavePeriod = 1 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/autoencoder_xception/"

shuffle_buffer_size = 10000
num_parallel_calls = 6
num_parallel_readers = 6
prefetch_buffer_size = 64

#batch_size = 8 #Batch size to use during training
num_epochs = 1000000 #Dataset repeats indefinitely

logDir = "C:/dump/train/"
log_file = model_dir+"log.txt"
log_every = 1 #Log every _ examples
cumProbs = np.array([]) #Indices of the distribution plus 1 will be correspond to means

numMeans = 64
scaleMean = 4 #Each means array index increment corresponds to this increase in the mean
numDynamicGrad = 10 # Number of gradients to calculate for each possible mean when dynamically updating training
lossSmoothingBoxcarSize = 5

#Dimensions of images in the dataset
height = width = 2048
channels = 1 #Greyscale input image

#Sidelength of images to feed the neural network
cropsize = 1024
height_crop = width_crop = cropsize

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

filters00 = 32
filters01 = 64
filters1 = 128
filters2 = 256
filters3 = 512
filters4 = 728
filters5 = 1024
filters6 = 1536
numMiddleXception = 16

##Modified aligned xception
def architecture(img, mode=None):
    """Atrous convolutional encoder-decoder noise-removing network"""

    #phase = mode == tf.estimator.ModeKeys.TRAIN #phase is true during training
    concat_axis = 3

    ##Reusable blocks

    def conv_block(input, filters, phase=phase):
        """
        Convolution -> batch normalisation -> leaky relu
        phase defaults to true, meaning that the network is being trained
        """

        conv_block = tf.layers.conv2d(inputs=input,
            filters=filters,
            kernel_size=3,
            padding="SAME",
            activation=tf.nn.relu)

        #conv_block = tf.contrib.layers.batch_norm(
        #    conv_block,
        #    center=True, scale=True,
        #    is_training=phase)

        #conv_block = tf.nn.leaky_relu(
        #    features=conv_block,
        #    alpha=0.2)
        #conv_block = tf.nn.relu(conv_block)

        return conv_block

    def aspp_block(input, phase=phase):
        """
        Atrous spatial pyramid pooling
        phase defaults to true, meaning that the network is being trained
        """

        #Convolutions at multiple rates
        conv1x1 = tf.layers.conv2d(inputs=input,
            filters=aspp_filters,
            kernel_size=1,
            padding="same",
            activation=tf.nn.relu,
            name="1x1")
        #conv1x1 = tf.contrib.layers.batch_norm(
        #    conv1x1,
        #    center=True, scale=True,
        #    is_training=phase)

        conv3x3_rateSmall = tf.layers.conv2d(inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateSmall,
            activation=tf.nn.relu,
            name="lowRate")
        #conv3x3_rateSmall = tf.contrib.layers.batch_norm(
        #    conv3x3_rateSmall,
        #    center=True, scale=True,
        #    is_training=phase)

        conv3x3_rateMedium = tf.layers.conv2d(inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateMedium,
            activation=tf.nn.relu,
            name="mediumRate")
        #conv3x3_rateMedium = tf.contrib.layers.batch_norm(
        #    conv3x3_rateMedium,
        #    center=True, scale=True,
        #    is_training=phase)

        conv3x3_rateLarge = tf.layers.conv2d(inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateLarge,
            activation=tf.nn.relu,
            name="highRate")
        #conv3x3_rateLarge = tf.contrib.layers.batch_norm(
        #    conv3x3_rateLarge,
        #    center=True, scale=True,
        #    is_training=phase)

        #Image-level features
        pooling = tf.nn.pool(input=input,
            window_shape=(2,2),
            pooling_type="AVG",
            padding="SAME",
            strides=(2, 2))
        #Use 1x1 convolutions to project into a feature space the same size as
        #the atrous convolutions'
        pooling = tf.layers.conv2d(
            inputs=pooling,
            filters=aspp_filters,
            kernel_size=1,
            padding="SAME",
            name="imageLevel")
        pooling = tf.image.resize_images(pooling, [64, 64])
        #pooling = tf.contrib.layers.batch_norm(
        #    pooling,
        #    center=True, scale=True,
        #    is_training=phase)

        #Concatenate the atrous and image-level pooling features
        concatenation = tf.concat(values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
            axis=concat_axis)

        #Reduce the number of channels
        reduced = tf.layers.conv2d(
            inputs=concatenation,
            filters=aspp_filters,
            kernel_size=1,
            padding="SAME")

        return reduced


    def strided_conv_block(input, filters, stride, rate=1, phase=phase):
        
        return slim.separable_convolution2d(inputs=input,
            num_outputs=filters,
            kernel_size=3,
            depth_multiplier=1,
            stride=stride,
            padding='SAME',
            data_format='NHWC',
            rate=rate,
            activation_fn=tf.nn.relu,
            normalizer_fn=None,
            normalizer_params=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=None,
            biases_initializer=tf.zeros_initializer(),
            biases_regularizer=None,
            reuse=None,
            variables_collections=None,
            outputs_collections=None,
            trainable=True,
            scope=None)

    def deconv_block(input, filters, phase=phase):
        '''Transpositionally convolute a feature space to upsample it'''
        
        #Residual
        residual = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=1,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)

        #Main flow
        main_flow = strided_conv_block(
            input=input,
            filters=filters,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters,
            stride=1)
        main_flow = tf.layers.conv2d_transpose(
            inputs=main_flow,
            filters=filters,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)

        return deconv_block + residual

    def xception_entry_flow(input):

        #Entry flow 0
        entry_flow = tf.layers.conv2d(
            inputs=input,
            filters=filters00,
            kernel_size=3,
            strides = 2,
            padding="SAME",
            activation=tf.nn.relu)
        entry_flow = tf.layers.conv2d(
            inputs=entry_flow,
            filters=filters01,
            kernel_size=3,
            padding="SAME",
            activation=tf.nn.relu)

        #Residual 1
        residual1 = tf.layers.conv2d(
            inputs=entry_flow,
            filters=filters1,
            kernel_size=1,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
       
        #Main flow 1
        main_flow1 = strided_conv_block(
            input=entry_flow,
            filters=filters1,
            stride=1)
        main_flow1 = strided_conv_block(
            input=main_flow1,
            filters=filters1,
            stride=1)
        main_flow1_strided = strided_conv_block(
            input=main_flow1,
            filters=filters1,
            stride=2)

        residual_connect1 = main_flow1_strided + residual1

        #Residual 2
        residual2 = tf.layers.conv2d(
            inputs=residual_connect1,
            filters=filters2,
            kernel_size=1,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
       
        #Main flow 2
        main_flow2 = strided_conv_block(
            input=main_flow1,
            filters=filters2,
            stride=1)
        main_flow2 = strided_conv_block(
            input=main_flow2,
            filters=filters2,
            stride=1)
        main_flow2_strided = strided_conv_block(
            input=main_flow2,
            filters=filters2,
            stride=2)

        residual_connect2 = main_flow2_strided + residual2

        #Residual 3
        residual3 = tf.layers.conv2d(
            inputs=residual_connect2,
            filters=filters3,
            kernel_size=1,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
       
        #Main flow 3
        main_flow3 = strided_conv_block(
            input=residual_connect2,
            filters=filters3,
            stride=1)
        main_flow3 = strided_conv_block(
            input=main_flow3,
            filters=filters3,
            stride=1)
        main_flow3_strided = strided_conv_block(
            input=main_flow3,
            filters=filters3,
            stride=2)

        residual_connect3 = main_flow3_strided + residual3

        #Residual 4
        residual4 = tf.layers.conv2d(
            inputs=residual_connect3,
            filters=filters4,
            kernel_size=1,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
       
        #Main flow 4
        main_flow4 = strided_conv_block(
            input=main_flow3,
            filters=filters4,
            stride=1)
        main_flow4 = strided_conv_block(
            input=main_flow4,
            filters=filters4,
            stride=1)
        main_flow4_strided = strided_conv_block(
            input=main_flow4,
            filters=filters4,
            stride=2)

        residual_connect4 = main_flow4_strided + residual4

        return residual_connect4

    def xception_middle_block(input):
        
        main_flow = strided_conv_block(
            input=input,
            filters=filters4,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters4,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters4,
            stride=1)

        return main_flow + residual


    def xception_exit_flow(input):

        #Residual
        residual = tf.layers.conv2d(
            inputs=input,
            filters=filters5,
            kernel_size=1,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)

        #Main flow
        main_flow = main_flow = strided_conv_block(
            input=minput,
            filters=filters5,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters5,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters5,
            stride=2)

        #Residual connection
        main_flow = main_flow + residual

        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters6,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters6,
            stride=1,
            rate=2)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters6,
            stride=1)

        return main_flow


    def autoencoding_decoder(input):

        #8x8
        decoding = tf.reshape(input, [-1, decode_size1, decode_size1, decode_channels1])
        decoding = conv_block(decoding, features = decode_channels1)
        decoding = conv_block(decoding, features = decode_channels1)
        decoding = conv_block(decoding, features = decode_channels1)

        #16x16
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decod_channels1,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, features = decode_channels2)
        decoding = conv_block(decoding, features = decode_channels2)
        decoding = conv_block(decoding, features = decode_channels2)

        #32x32
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decod_channels2,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, features = decode_channels3)
        decoding = conv_block(decoding, features = decode_channels3)
        decoding = conv_block(decoding, features = decode_channels3)

        #64x64
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decod_channels3,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, features = decode_channels4)
        decoding = conv_block(decoding, features = decode_channels4)

        #128x128
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decod_channels4,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, features = decode_channels5)
        decoding = conv_block(decoding, features = decode_channels5)

        #256x256
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decod_channels5,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, features = decode_channels6)
        decoding = conv_block(decoding, features = decode_channels6)

        #512x512
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decod_channels6,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, features = decode_channels7)
        decoding = conv_block(decoding, features = decode_channels7)

        #1024x1024
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decod_channels7,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, features = decode_channels8)
        decoding = conv_block(decoding, 1)

        return decoding

    '''Model building'''
    input_layer = tf.reshape(img, [-1, cropsize, cropsize, channels])

    #Build Xception
    main_flow = xception_entry_flow(input_layer)

    for _ in range(numMiddleXception):
        main_flow = xception_middle_block(main_flow)

    main_flow = xception_exit_flow(main_flow)

    fc = tf.contrib.layers.fully_connected(
        input,
        fc_features1)

    output = autoencoding_decoder(fc)

    #Image values will be between 0 and 1
    output = tf.clip_by_value(output,
        clip_value_min=0.0,
        clip_value_max=1.0)

    if phase: #Calculate loss during training
        loss = 1.0-tf_ssim(output, input_layer)
    else:
        loss = -1

    return loss, output


def load_image(addr, resizeSize=None, imgType=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""
    
    img = imread(addr, mode='F')
    if resizeSize:
        img = cv2.resize(img, resizeSize, interpolation=cv2.INTER_CUBIC)
    img = img.astype(imgType)

    return img


def scale0to1(img):
    """Rescale image between 0 and 1"""

    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)


def flip_rotate(img):
    """Applies a random flip || rotation to the image, possibly leaving it unchanged"""

    choice = int(8*np.random.rand())
    
    if choice == 0:
        return img
    if choice == 1:
        return np.rot90(img, 1)
    if choice == 2:
        return np.rot90(img, 2)
    if choice == 3:
        return np.rot90(img, 3)
    if choice == 4:
        return np.flip(img, 0)
    if choice == 5:
        return np.flip(img, 1)
    if choice == 6:
        return np.flip(np.rot90(img, 1), 0)
    if choice == 7:
        return np.flip(np.rot90(img, 1), 1)


def preprocess(img):
    """
    Threshold the image to remove dead or very bright pixels.
    Then crop a region of the image of a random size and resize it.
    """

    size = int(cropsize + np.random.rand()*(height-cropsize))
    topLeft_x = int(np.random.rand()*(height-size))
    topLeft_y = int(np.random.rand()*(height-size))

    crop = img[topLeft_y:(topLeft_y+cropsize), topLeft_x:(topLeft_x+cropsize)]

    resized = cv2.resize(crop, (cropsize, cropsize), interpolation=cv2.INTER_AREA)

    resized[np.isnan(resized)] = 0.5
    resized[np.isinf(resized)] = 0.5

    return scale0to1(flip_rotate(resized))


def parser(record):
    """Parse files and generate lower quality images from them"""

    img = load_image(record)
    img = preprocess(img)

    return img


def input_fn(dir):
    """Create a dataset from a list of filenames"""

    dataset = tf.data.Dataset.list_files(dir+"*.tif")
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(
        lambda file: tuple(tf.py_func(parser, [file], [tf.float32])),
        num_parallel_calls=num_parallel_calls)
    #dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    dataset = dataset.repeat(num_epochs)
    
    iter = dataset.make_one_shot_iterator()

    img = iter.get_next()

    return img



def main(unused_argv=None):

    temp = set(tf.all_variables())

    log = open(log_file, 'a')

    #with tf.device("/gpu:0"):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
    with tf.control_dependencies(update_ops):

        img = input_fn(trainDir)

        loss, prediction = architecture(img, tf.estimator.ModeKeys.TRAIN)
        train_op = tf.train.AdamOptimizer().minimize(loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.7

        #saver = tf.train.Saver(max_to_keep=-1)

        tf.add_to_collection("train_op", train_op)
        tf.add_to_collection("update_ops", update_ops)
        with tf.Session(config=config) as sess: #Alternative is tf.train.MonitoredTrainingSession()

            init = tf.global_variables_initializer()

            sess.run(init)
            sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

            train_writer = tf.summary.FileWriter( logDir, sess.graph )

            #Set up mean probabilities to be dynamically adjusted during training
            probs = np.ones(numMeans, dtype=np.float32)
            losses0 = np.empty([])
            global cumProbs
            cumProbs = np.cumsum(probs)
            cumProbs /= np.max(cumProbs)

            #print(tf.all_variables())

            counter = 0
            cycleNum = 0
            while True:
                cycleNum += 1
                #Train for a couple of hours
                time0 = time.time()
                while time.time()-time0 < modelSavePeriod:
                    counter += 1

                    #merge = tf.summary.merge_all()

                    _, loss_value = sess.run([train_op, loss])
                    print("Iter: {}, Loss: {:.6f}".format(counter, loss_value))
                    log.write("Iter: {}, Loss: {:.6f}".format(counter, loss_value))
                    #train_writer.add_summary(summary, counter)

                #Save the model
                #saver.save(sess, save_path=model_dir+"model", global_step=counter)
                tf.saved_model.simple_save(
                    session=sess,
                    export_dir=model_dir+"model-"+str(counter)+"/",
                    inputs={"img": img},
                    outputs={"prediction": prediction})

                #predict_fn = tf.contrib.predictor.from_saved_model(model_dir+"model-"+str(counter)+"/")

                #loaded_img = imread("E:/stills_hq/reaping1.tif", mode='F')
                #loaded_img = scale0to1(cv2.resize(loaded_img, (cropsize, cropsize), interpolation=cv2.INTER_AREA))
                #cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
                #cv2.imshow("dfsd", loaded_img)
                #cv2.waitKey(0)

                #prediction1 = predict_fn({"lq": loaded_img})

                #cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
                #cv2.imshow("dfsd", prediction1['prediction'].reshape(cropsize, cropsize))
                #cv2.waitKey(0)

                #Evaluate the model and use the results to dynamically adjust the training process
                losses = np.zeros(numMeans, dtype=np.float32)
                for i in range(numMeans):
                    for _ in range(numDynamicGrad):
                        losses[i] += sess.run(loss)
                        print(i, losses[i])
                    losses[i] /= numDynamicGrad

                np.save(model_dir+"losses-"+str(counter), losses)

                #cumProbs = get_training_probs(losses0, losses)
                losses0 = losses
    return 

if __name__ == "__main__":
    tf.app.run()

