from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

##Testing variables that desctibe the training dataset

imgOrigSize = [2048, 2048, 1] #Size of images

#Addresses to save the TFRecords files to
train_filename = '//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/stills/train.tfrecords'
val_filename = '//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/stills/val.tfrecords'
test_filename = '//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/stills/test.tfrecords'

#Initial location of dat
dataLoc = '//flexo.ads.warwick.ac.uk/shared39/EOL2100/2100/Users/Jeffrey-Ede/datasets/stills/'

## Shuffle data addresses and separate data into training, validation and test sets

import os
from random import shuffle

shuffle_data = True  # shuffle the addresses before saving

#Read addresses from the 'train' folder
addrs = os.listdir(dataLoc)
addrs = [dataLoc+file for file in addrs if '.tif' in file]

#Shuffle data
if shuffle_data:
    shuffle(addrs)
    
#Divide the hata into 70% train, 15% validation and 15% test
train_addrs = addrs[0:int(0.7*len(addrs))]
val_addrs = addrs[int(0.7*len(addrs)):int(0.85*len(addrs))]
test_addrs = addrs[int(0.85*len(addrs)):]

##Image loading function

import numpy as np
import tensorflow as tf
import sys
import cv2
from scipy.misc import imread

def load_image(addr, resizeSize=None, imgType=np.float32):
    #Read an image and make sure it is of the correct type. Optionally resize it
    
    img = imread(addr, mode='F')
    
    if resizeSize:
        img = cv2.resize(img, resizeSize, interpolation=cv2.INTER_CUBIC)
    
    img = img.astype(imgType)

    return img

##Write data to tfrecord

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

print(len(train_addrs))

## 1) Training data...

#Open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(train_addrs)):
    #Print how many images are saved every 1000 images
    if not i % 1000:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
        
    #Load the image
    img = load_image(train_addrs[i])

    #Create a feature
    feature = {'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    #Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    #Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()

## 2) Validation data

#Open the TFRecords file
writer = tf.python_io.TFRecordWriter(val_filename)

for i in range(len(val_addrs)):
    #Print how many images are saved every 1000 images
    if not i % 1000:
        print('Val data: {}/{}'.format(i, len(val_addrs)))
        #sys.stdout.flush()
    
    #Load the image
    img = load_image(val_addrs[i])
    
    #Create a feature
    feature = {'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    
    #Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()

## 3) Test data

#Open the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)

for i in range(len(test_addrs)):
    #Print how many images are saved every 1000 images
    if not i % 1000:
        print('Test data: {}/{}'.format(i, len(test_addrs)))
        #sys.stdout.flush()

    #Load the image
    img = load_image(test_addrs[i])
    
    #Create a feature
    feature = {'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    
    #Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    #Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()

##Read a TFRecords file

import matplotlib.pyplot as plt
data_path = 'train.tfrecords'  #Address to save the hdf5 file

with tf.Session() as sess:
    feature = {'train/image': tf.FixedLenFeature([], tf.string)}

    #Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)
    
    #Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    #Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['train/image'], tf.float32)
    
    #Reshape image data into the original shape
    image = tf.reshape(image, imgOrigShape)
    
    # Any preprocessing here ...
    
    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)

#Initialize all global and local variables
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

#Create a coordinator and run all QueueRunner objects
coord = tf.train.Coordinator()

threads = tf.train.start_queue_runners(coord=coord)
for batch_index in range(5):
    img = sess.run([images])
    img = img.astype(np.float32)
    
    for j in range(6):
        plt.subplot(2, 3, j+1)
        plt.imshow(img[j, ...])

    plt.show()

#Stop the threads
coord.request_stop()
    
# Wait for threads to stop
coord.join(threads)
sess.close()