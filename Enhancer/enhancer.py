from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

rows = cols = 256 # For simplicity during development
channels = 1 #Greyscale input image

features1 = 16 #Number of features to use for initial convolution
features2 = 2*features1 #Number of features after 2nd convolution
features3 = 2*features2 #Number of features after 3rd convolution
features4 = 2*features3 #Number of features after 4th convolution
aspp_filters = features4 #Number of features for atrous convolutional spatial pyramid pooling

aspp_rateSmall = 4
aspp_rateMedium = 8
aspp_rateLarge = 12

tfrecords_train_filenames = ["location..."]
tfrecords_val_filenames = ["location..."]
tfrecords_test_filenames = ["location..."]

batch_size = 1 #Batch size to use during training

def cnn_model_fn(features, labels, mode):
    '''Atrous convolutional encoder-decoder noise-removing network'''
    
    '''Helper functions'''
    aspp_block(input):
        '''Atrous spatial pyramid pooling'''

        #Convolutions at multiple rates
        conv1x1 = tf.layers.conv2d(
        inputs=input,
        filters=aspp_filters,
        kernel_size=1,
        padding="same")

        conv3x3_rateSmall = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateSmall)

        conv3x3_rateMedium = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateMedium)

        conv3x3_rateLarge = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateLarge)

        #Image-level features
        pooling = tf.nn.pool(
            input=input,
            window_shape=(2,2),
            pooling_type="AVG",
            padding="same")

        #Concatenate the atrous and image-level features
        concatenation = tf.concat(
        values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
        axis=axis)

        #Reduce the number of channels
        reduced = tf.layers.conv2d( #Not sure if this is the correct way to reshape...
            inputs=input,
            filters=aspp_filters,
            kernel_size=1,
            padding="same")

        return reduced

    '''Model building'''
    input_layer = tf.reshape(features["x"], [-1, rows, cols, 1])

    #Encoder
    cnn1 = tf.nn.convolution(
        input=input_layer,
        filter=features1,
        padding="same",
        activation=tf.nn.relu)

    cnn1_strided = tf.layers.conv2d(
        inputs=cnn1,
        filters=features1,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    cnn2 = tf.nn.convolution(
        input=cnn1_strided,
        filter=features2,
        padding="same",
        activation=tf.nn.relu)

    cnn2_strided = tf.layers.conv2d(
        inputs=cnn2,
        filters=features2,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    cnn3 = tf.nn.convolution(
        input=cnn2_strided,
        filter=features3,
        padding="same",
        activation=tf.nn.relu)

    cnn3_strided = tf.layers.conv2d(
        inputs=cnn3,
        filters=features3,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    cnn4 = tf.nn.convolution(
        input=cnn3_strided,
        filter=features4,
        padding="same",
        activation=tf.nn.relu)

    cnn4_strided = tf.layers.conv2d(
        inputs=cnn4,
        filters=features3,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    #Atrous spatial pyramid pooling
    aspp = aspp_block(cnn4_strided)

    #Upsample the semantics by a factor of 4
    #upsampled_aspp = tf.image.resize_bilinear(
    #    images=aspp,
    #    tf.shape(aspp)[1:3],
    #    align_corners=True)

    '''Deconvolute the semantics'''
    concat3 = tf.concat(
        values=[cnn3, aspp],
        axis=axis)

    deconv3 = tf.layers.conv2d_transpose(
        inputs=concat3,
        filters=features3,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    deconv3_strided = tf.layers.conv2d_transpose(
        inputs=concat2,
        filters=features2,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=tf.nn.relu)

    concat2 = tf.concat(
        values=[cnn2, deconv3_strided],
        axis=axis)

    deconv2 = tf.layers.conv2d_transpose(
        inputs=concat2,
        filters=features2,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    deconv2_strided = tf.layers.conv2d_transpose(
        inputs=concat2,
        filters=features1,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=tf.nn.relu)

    concat1 = tf.concat(
        values=[cnn1, deconv2_strided],
        axis=axis)

    deconv1 = tf.layers.conv2d_transpose(
        inputs=concat1,
        filters=features1,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    deconv_final = tf.layers.conv2d_transpose(
        inputs=deconv1,
        filters=1,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    #Residually connect the input to the output
    output = input_layer + deconv_final

    '''Evaluation'''
    predictions = {
        "output": output
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels,
                                        predictions=output)

    '''Training'''
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    '''Evaluation'''
    eval_metric_ops = {
        "loss": loss }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def dataset_input_fn():
    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)

    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
    # protocol buffer, and perform any additional per-record preprocessing.
    def parser(record):
        keys_to_features = {
            "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
            "date_time": tf.FixedLenFeature((), tf.int64, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64,
                                        default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data
        image = tf.image.decode_jpeg(parsed["image_data"])
        image = tf.reshape(image, [299, 299, 1])
        label = tf.cast(parsed["label"], tf.int32)

        return {"image_data": image, "date_time": parsed["date_time"]}, label

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    features, labels = iterator.get_next()

    return features, labels

def main(unused_argv):

    def _deteriorate_img(img):
    '''Create a low quality image from high quality image by using a smaller number of counts'''

    return deterioration

    def _deterioration_probs(hist):
        '''Calculate training data deterioration amount probabilities'''

        return probs

    def parser(record):
        '''Parse a TFRecord and return a training example'''
        keys_to_features = {
            "img": tf.FixedLenFeature((), tf.string, default_value=""),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        
        img = tf.reshape(image, [2048, 2048, 1])

        return deterioration, img

    #Classify training, validation and test data
    train_data = tf.data.TFRecordDataset(tfrecords_train_filenames)
    val_data = tf.data.TFRecordDataset(tfrecords_val_filenames)
    test_data = tf.data.TFRecordDataset(tfrecords_test_filenames)

    #Break the dataset into shards and iteratively train on the shards
    for shard_idx in range(num_shards):
        train_shard = train_data.shard(num_shards=num_shards, index=shard_idx)

    #Process records to get training examples
    train_data.map(map_func=parser)
    val_data.map(map_func=parser)
    test_data.map(map_func=parser)

    #Batch data
    train_data.batch(batch_size)
    val_data.batch(batch_size)
    test_data.batch(batch_size)

    # A feedable iterator is defined by a handle placeholder and its structure. We
    # could use the `output_types` and `output_shapes` properties of either
    # `training_dataset` or `validation_dataset` here, because they have
    # identical structure.
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_data.output_types, train_data.output_shapes)
    next_element = iterator.get_next()

    #Create feedable iterators
    train_iter = train_data.make_one_shot_iterator()
    val_iter = val_data.make_initializable_iterator()
    test_iter = test_data.make_initializable_iterator()

    #Create session handles that can be evaluated to yield examples
    train_handle = sess.run(train_iter.string_handle())
    val_handle = sess.run(val_iter.string_handle())
    test_handle = sess.run(test_iter.string_handle())

    ## Loop forever, alternating between training and validation.
    #while True:
    #    # Run 200 steps using the training dataset. Note that the training dataset is
    #    # infinite, and we resume from where we left off in the previous `while` loop
    #    # iteration.
    #    for _ in range(200):
    #        sess.run(next_element, feed_dict={handle: train_handle})

    #    # Run one pass over the validation dataset.
    #    sess.run(val_iter.initializer)
    #    for _ in range(50):
    #        sess.run(next_element, feed_dict={handle: val_handle})

    #batched_dataset = dataset.batch(batch_size)

    #iterator = batched_dataset.make_one_shot_iterator()
    #next_element = iterator.get_next()

    return 

if __main__ = "__main__":
    tf.app.run()