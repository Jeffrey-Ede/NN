from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

#rows = 667
#cols = 672
rows = cols = 256 # For simplicity during development

channels = 2 #Amplitude and phase

feature_space_size = 32 #Number of features that the initial convolutional layer projects onto

def cnn_model_fn(features, labels, mode):
    """Holographic reconstruction neural network"""

    '''Helper functions'''
    def create_residual_block(input):
        '''Residually concatenates the block input after flowing it through 2 convolutional layers'''

        residual_block = tf.layers.conv2d(
            inputs=input,
            filters=feature_space_size,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)
        
        residual_block = tf.layers.conv2d(
            inputs=residual_block,
            filters=feature_space_size,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)

        residual_block = input + residual_block #Residual connection

        return residual_block

    def create_blockchain(input):
        '''Define the blockchain used for each of the downsamplings'''

        blockchain = tf.layers.conv2d(
            inputs=input,
            filters=feature_space_size,
            kernel_size=3,
            padding="same",
            activation=None)

        blockchain = create_residual_block(blockchain)
        blockchain = create_residual_block(blockchain)
        blockchain = create_residual_block(blockchain)
        blockchain = create_residual_block(blockchain)

        return blockchain

    def create_upsampling_block(input, new_rows, new_cols):
        '''Use a convolutional layer to project the input into a larger feature space that can be reshaped to upsample the input'''

        upsampling_block = tf.layers.conv2d(
            inputs=input,
            filters=4*feature_space_size,
            kernel_size=3,
            padding="same",
            activation=tf.nn.relu)

        upsampling_block = tf.reshape(
            tensor=upsampling_block,
            shape=(int(new_rows), int(new_cols), feature_space_size))

        return upsampling_block

    '''Model building'''
    input_layer = tf.reshape(features["x"], [-1, rows, cols, 1])

    #Downsample the image by various amounts to process features on various scales
    d1 = tf.AveragePooling2D(
        inputs=input_layer,
        pool_size=1,
        strides=1,
        padding="valid")
    
    d2 = tf.AveragePooling2D(
        inputs=input_layer,
        pool_size=2,
        strides=2,
        padding="valid")

    d4 = tf.AveragePooling2D(
        inputs=input_layer,
        pool_size=4,
        strides=4,
        padding="valid")

    d8 = tf.AveragePooling2D(
        inputs=input_layer,
        pool_size=8,
        strides=8,
        padding="valid")

    #Flow downsamplings through residually connected blockchains
    d1 = create_blockchain(d1)
    d2 = create_blockchain(d2)
    d4 = create_blockchain(d4)
    d8 = create_blockchain(d8)

    #Upsample the downsampled flows
    d2 = create_upsampling_block(d2, rows, cols)

    d4 = create_upsampling_block(d4, rows/2, cols/2)
    d4 = create_upsampling_block(d4, rows, cols)

    d8 = create_upsampling_block(d8, rows/4, cols/4)
    d8 = create_upsampling_block(d8, rows/2, cols/2)
    d8 = create_upsampling_block(d8, rows, cols)

    #Apply convolutional layers and concatenate the outputs
    d1 = tf.layers.conv2d(
        inputs=d1,
        filters=feature_space_size,
        kernel_size=3,
        padding="same")

    d2 = tf.layers.conv2d(
        inputs=d2,
        filters=feature_space_size,
        kernel_size=3,
        padding="same")

    d4 = tf.layers.conv2d(
        inputs=d4,
        filters=feature_space_size,
        kernel_size=3,
        padding="same")

    d8 = tf.layers.conv2d(
        inputs=d8,
        filters=feature_space_size,
        kernel_size=3,
        padding="same")

    #Concatenate the 4 flows
    concatenation = tf.concat(
        values=[d1, d2, d4, d8],
        axis=axis)

    concatenation = tf.layers.conv2d(
        inputs=concatenation,
        filters=2,
        kernel_size=3,
        padding="same")

    loss = tf.losses.mean_squared_error(
        labels=labels,
        predictions=predictions)

    #Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        
        train_op = optimizer.minimise(
            loss=loss,
            global_step=tf.train.get_global_step())
        
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #Add evaluation metrics (for EVAL mode)
    # Add these later...


    return 1

def main(unused_argv):
    #Load training and evaluation data...


if __main__ = "__main__":
    tf.app.run()