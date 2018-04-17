from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

import time

slim = tf.contrib.slim #For depthwise separable strided atrous convolutions

tf.logging.set_verbosity(tf.logging.DEBUG)

features1 = 32 #Number of features to use for initial convolution
features2 = 2*features1 #Number of features after 2nd convolution
features3 = 3*features1 #Number of features after 3rd convolution
features4 = 4*features1 #Number of features after 4th convolution
aspp_filters = features4 #Number of features for atrous convolutional spatial pyramid pooling

aspp_rateSmall = 6
aspp_rateMedium = 12
aspp_rateLarge = 18

trainDir = "F:/stills_hq/train/"
valDir = "F:/stills_hq/val/"
testDir = "F:/stills_hq/test/"

modelSavePeriod = 0.005 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "E:/models/noise1/"

shuffle_buffer_size = 5000
num_parallel_calls = 7
prefetch_buffer_size = 1

batch_size = 8 #Batch size to use during training
num_epochs = -1 #Dataset repeats indefinitely

logDir = "C:/dump/train/"
log_every = 1 #Log every _ examples
cumProbs = np.array([]) #Indices of the distribution plus 1 will be correspond to means

#Remove extreme intensities
removeLower = 0.01
removeUpper = 0.01

numMeans = 10
numDynamicGrad = 1 # Number of gradients to calculate for each possible mean when dynamically updating training
lossSmoothingBoxcarSize = 7

#Dimensions of images in the dataset
height = width = 2048
channels = 1 #Greyscale input image

#Sidelength of images to feed the neural network
cropsize = 512
height_crop = width_crop = cropsize

## Initial idea: aspp, batch norm + Leaky RELU, residual connection and lower feature numbers
def architecture(lq, img, mode):
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
            padding="SAME")

        conv_block = tf.contrib.layers.batch_norm(
            conv_block, 
            center=True, scale=True, 
            is_training=phase)

        #conv_block = tf.nn.leaky_relu(
        #    features=conv_block,
        #    alpha=0.2)
        conv_block = tf.nn.relu(conv_block)

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


    def strided_conv_block(input, filters, stride, phase=phase):
        
        return slim.separable_convolution2d(
            inputs=input,
            num_outputs=filters,
            kernel_size=3,
            depth_multiplier=1,
            stride=stride,
            padding='SAME',
            data_format='NHWC',
            rate=1,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
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

        #deconv_block = tf.nn.leaky_relu(
        #    features=deconv_block,
        #    alpha=0.2)
        deconv_block = tf.nn.relu(deconv_block)

        return deconv_block

    '''Model building'''
    input_layer = tf.reshape(lq, [-1, cropsize, cropsize, channels])

    #Encoding block 1
    cnn1_last = conv_block(
        input=input_layer, 
        filters=features1)
    cnn1_strided = strided_conv_block(
        input=cnn1_last,
        filters=features1,
        stride=2)

    #Encoding block 2
    cnn2_last = conv_block(
        input=cnn1_strided,
        filters=features2)
    cnn2_strided = strided_conv_block(
        input=cnn2_last,
        filters=features2,
        stride=2)

    #Encoding block 3
    #cnn3 = conv_block(
    #    input=cnn2_strided,
    #    filters=features3)
    #cnn3_last = conv_block(
    #    input=cnn3,
    #    filters=features3)
    cnn3_last = conv_block(
        input=cnn2_strided,
        filters=features3)
    cnn3_strided = strided_conv_block(
        input=cnn3_last,
        filters=features3,
        stride=2)

    #Encoding block 4
    #cnn4 = conv_block(
    #    input=cnn3_strided,
    #    filters=features4)
    #cnn4_last = conv_block(
    #    input=cnn4,
    #    filters=features4)
    cnn4_last = conv_block(
        input=cnn3_strided,
        filters=features4)

    #cnn4_strided = split_separable_conv2d(
    #    inputs=cnn4_last,
    #    filters=features4,
    #    rate=2,
    #    stride=2)

    ##Atrous spatial pyramid pooling

    aspp = aspp_block(cnn4_last)

    #Upsample the semantics by a factor of 4
    #upsampled_aspp = tf.image.resize_bilinear(
    #    images=aspp,
    #    tf.shape(aspp)[1:3],
    #    align_corners=True)

    #Decoding block 1 (deepest)
    deconv4 = conv_block(aspp, features4)
    #deconv4 = conv_block(deconv4, features4)
    
    #Decoding block 2
    deconv4to3 = deconv_block(deconv4, features4)
    concat3 = tf.concat(
        values=[deconv4to3, cnn3_last],
        axis=concat_axis)
    deconv3 = conv_block(concat3, features3)
    #deconv3 = conv_block(deconv3, features3)

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
    output = deconv_final#+input_layer

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

def scale0to1(img):
    """Rescale image between 0 and 1"""

    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def gen_lq(img, mean):
    '''Generate low quality image'''

    #Ensure that the seed is random
    np.random.seed(int(np.random.rand()*(2**32-1)))

    #Adjust the image scale so that the image has the correct average counts
    lq = np.random.poisson(mean * (img / np.mean(img)))
    
    return scale0to1(lq)

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

    sorted = np.sort(img, axis=None)
    min = sorted[int(removeLower*sorted.size)]
    max = sorted[int((1.0-removeUpper)*sorted.size)]

    size = int(cropsize + np.random.rand()*(height-cropsize))
    topLeft_x = int(np.random.rand()*(height-size))
    topLeft_y = int(np.random.rand()*(height-size))

    crop = np.clip(img[topLeft_y:(topLeft_y+cropsize), topLeft_x:(topLeft_x+cropsize)], min, max)

    resized = cv2.resize(crop, (cropsize, cropsize), interpolation=cv2.INTER_AREA)

    return scale0to1(flip_rotate(resized))

def get_mean(cumProbs):
    """Generate a mean from the cumulative probability distribution"""

    r = np.random.rand()
    idx = next(idx for idx, value in enumerate(cumProbs) if value >= r)

    if idx:
        return idx + r - cumProbs[idx-1]
    else:
        return 1

def parser(record):
    """Parse files and generate lower quality images from them"""

    img = load_image(record)
    img = preprocess(img)

    mean = get_mean(cumProbs)
    lq = gen_lq(img, mean)

    return lq, img

def input_fn(dir):
    """Create a dataset from a list of filenames"""

    dataset = tf.data.Dataset.list_files(dir+"*.tif")
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(
        lambda file: tuple(tf.py_func(parser, [file], [tf.float32, tf.float32])),
        num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    dataset = dataset.repeat(num_epochs)
    
    iter = dataset.make_one_shot_iterator()
        
    lq, img = iter.get_next()

    return lq, img

def movingAverage(values, window):

    weights = np.repeat(1.0, window)/window
    ma = np.convolve(values, weights, 'same')

    return ma

def get_training_probs(losses0, losses1):
    """
    Returns cumulative probabilities of means being selected for loq-quality image syntheses
    losses0 - previous losses (smoothed)
    losses1 - losses after the current training run
    """

    diffs = movingAverage(losses1, lossSmoothingBoxcarSize) - losses0
    diffs[diffs > 0] = 0
    cumDiffs = np.cumsum(diffs)
    cumProbs = cumDiffs / np.max(cumDiffs)

    return cumProbs.astype(np.float32)

def main(unused_argv=None):

    temp = set(tf.all_variables())

    #with tf.device("/gpu:0"):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
    with tf.control_dependencies(update_ops):

        lq, img = input_fn(trainDir)

        loss, prediction = architecture(lq, img, tf.estimator.ModeKeys.TRAIN)
        train_op = tf.train.AdamOptimizer().minimize(loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.7

        saver = tf.train.Saver(max_to_keep=-1)

        tf.add_to_collection("train_op", train_op)
        tf.add_to_collection("update_ops", update_ops)
        with tf.Session(config=config) as sess: #Alternative is tf.train.MonitoredTrainingSession()

            init = tf.global_variables_initializer()

            sess.run(init)
            sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

            train_writer = tf.summary.FileWriter( logDir, sess.graph )

            counter = 0
            saveNum = 1

            #Set up mean probabilities to be dynamically adjusted during training
            probs = np.ones(numMeans, dtype=np.float32)
            losses0 = probs*numDynamicGrad
            global cumProbs
            cumProbs = np.cumsum(probs)
            cumProbs /= np.max(cumProbs)

            #print(tf.all_variables())

            while True:
                #Train for a couple of hours
                time0 = time.time()
                while time.time()-time0 < modelSavePeriod:
                    counter += 1

                    merge = tf.summary.merge_all()

                    summary, _, loss_value = sess.run([merge, train_op, loss])
                    print("Iter: {}, Loss: {:.4f}".format(counter, loss_value))

                    train_writer.add_summary(summary, counter)

                #Save the model
                saver.save(sess, save_path=model_dir+"model", global_step=counter)
                saveNum += 1

                #Evaluate the model and use the results to dynamically adjust the training process
                losses = np.zeros(numMeans, dtype=np.float32)
                for i in range(numMeans):
                    for _ in range(numDynamicGrad):
                        losses[i] += sess.run(loss)
                        print(i, losses[i])
                    losses[i] /= numDynamicGrad

                cumProbs = get_training_probs(losses0, losses)
                losses0 = losses

    return 

if __name__ == "__main__":
    tf.app.run()
