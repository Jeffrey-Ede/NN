##USEFUL
#To get tensorboard:
#1) python "C:/Program Files (x86)/Microsoft Visual Studio/Shared/Python36_64/Lib/site-packages/tensorboard/main.py" --logdir=c/dump/train
#2) http://DESKTOP-SA1EVJV:6006 

## Initial model
def cnn_model_fn_initial(features, labels, mode):
    '''Atrous convolutional encoder-decoder noise-removing network'''
    
    '''Helper functions'''
    def aspp_block(input):
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


## Initial idea: aspp, batch norm + PRELU, no residual connection at end, lower feature numbers
def cnn_model_fn_idea1(features, labels, mode):
    '''Atrous convolutional encoder-decoder noise-removing network'''
    
    '''Helper functions'''
    def aspp_block(input):
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

### Initial model
#def cnn_model_fn_initialIdea(features, labels, mode):
#    '''Atrous convolutional encoder-decoder noise-removing network'''
    
#    '''Helper functions'''
#    def aspp_block(input):
#        '''Atrous spatial pyramid pooling'''

#        #Convolutions at multiple rates
#        conv1x1 = tf.layers.conv2d(
#        inputs=input,
#        filters=aspp_filters,
#        kernel_size=1,
#        padding="same")

#        conv3x3_rateSmall = tf.layers.conv2d(
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=3,
#            padding="same",
#            dilation_rate=aspp_rateSmall)

#        conv3x3_rateMedium = tf.layers.conv2d(
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=3,
#            padding="same",
#            dilation_rate=aspp_rateMedium)

#        conv3x3_rateLarge = tf.layers.conv2d(
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=3,
#            padding="same",
#            dilation_rate=aspp_rateLarge)

#        #Image-level features
#        pooling = tf.nn.pool(
#            input=input,
#            window_shape=(2,2),
#            pooling_type="AVG",
#            padding="same")

#        #Concatenate the atrous and image-level features
#        concatenation = tf.concat(
#        values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
#        axis=axis)

#        #Reduce the number of channels
#        reduced = tf.layers.conv2d( #Not sure if this is the correct way to reshape...
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=1,
#            padding="same")

#        return reduced

#    '''Model building'''
#    input_layer = tf.reshape(features["x"], [-1, rows, cols, 1])

#    #Encoder
#    cnn1 = tf.nn.convolution(
#        input=input_layer,
#        filter=features1,
#        padding="same",
#        activation=tf.nn.relu)

#    cnn1_strided = tf.layers.conv2d(
#        inputs=cnn1,
#        filters=features1,
#        kernel_size=3,
#        strides=2,
#        padding='same',
#        activation=tf.nn.relu)

#    cnn2 = tf.nn.convolution(
#        input=cnn1_strided,
#        filter=features2,
#        padding="same",
#        activation=tf.nn.relu)

#    cnn2_strided = tf.layers.conv2d(
#        inputs=cnn2,
#        filters=features2,
#        kernel_size=3,
#        strides=2,
#        padding='same',
#        activation=tf.nn.relu)

#    cnn3 = tf.nn.convolution(
#        input=cnn2_strided,
#        filter=features3,
#        padding="same",
#        activation=tf.nn.relu)

#    cnn3_strided = tf.layers.conv2d(
#        inputs=cnn3,
#        filters=features3,
#        kernel_size=3,
#        strides=2,
#        padding='same',
#        activation=tf.nn.relu)

#    cnn4 = tf.nn.convolution(
#        input=cnn3_strided,
#        filter=features4,
#        padding="same",
#        activation=tf.nn.relu)

#    cnn4_strided = tf.layers.conv2d(
#        inputs=cnn4,
#        filters=features3,
#        kernel_size=4,
#        strides=2,
#        padding='same',
#        activation=tf.nn.relu)

#    #Atrous spatial pyramid pooling
#    aspp = aspp_block(cnn4_strided)

#    #Upsample the semantics by a factor of 4
#    #upsampled_aspp = tf.image.resize_bilinear(
#    #    images=aspp,
#    #    tf.shape(aspp)[1:3],
#    #    align_corners=True)

#    '''Deconvolute the semantics'''
#    concat3 = tf.concat(
#        values=[cnn3, aspp],
#        axis=axis)

#    deconv3 = tf.layers.conv2d_transpose(
#        inputs=concat3,
#        filters=features3,
#        kernel_size=3,
#        padding="same",
#        activation=tf.nn.relu)

#    deconv3_strided = tf.layers.conv2d_transpose(
#        inputs=concat2,
#        filters=features2,
#        kernel_size=3,
#        strides=2,
#        padding="same",
#        activation=tf.nn.relu)

#    concat2 = tf.concat(
#        values=[cnn2, deconv3_strided],
#        axis=axis)

#    deconv2 = tf.layers.conv2d_transpose(
#        inputs=concat2,
#        filters=features2,
#        kernel_size=3,
#        padding="same",
#        activation=tf.nn.relu)

#    deconv2_strided = tf.layers.conv2d_transpose(
#        inputs=concat2,
#        filters=features1,
#        kernel_size=3,
#        strides=2,
#        padding="same",
#        activation=tf.nn.relu)

#    concat1 = tf.concat(
#        values=[cnn1, deconv2_strided],
#        axis=axis)

#    deconv1 = tf.layers.conv2d_transpose(
#        inputs=concat1,
#        filters=features1,
#        kernel_size=3,
#        padding="same",
#        activation=tf.nn.relu)

#    deconv_final = tf.layers.conv2d_transpose(
#        inputs=deconv1,
#        filters=1,
#        kernel_size=3,
#        padding="same",
#        activation=tf.nn.relu)

#    #Residually connect the input to the output
#    output = input_layer + deconv_final

#    '''Evaluation'''
#    predictions = {
#        "output": output
#    }

#    if mode == tf.estimator.ModeKeys.PREDICT:
#        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

#    #Calculate Loss (for both TRAIN and EVAL modes)
#    loss = tf.losses.mean_squared_error(labels=labels,
#                                        predictions=output)

#    '''Training'''
#    if mode == tf.estimator.ModeKeys.TRAIN:
#        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#        train_op = optimizer.minimize(
#            loss=loss,
#            global_step=tf.train.get_global_step())
#        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
#    '''Evaluation'''
#    eval_metric_ops = {
#        "loss": loss }
#    return tf.estimator.EstimatorSpec(
#        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


##Write data to tfrecord

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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

#def dataset_input_fn():
#    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
#    dataset = tf.data.TFRecordDataset(filenames)

#    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
#    # protocol buffer, and perform any additional per-record preprocessing.
#    def parser(record):
#        keys_to_features = {
#            "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
#        }
#        parsed = tf.parse_single_example(record, keys_to_features)
#        tf.decode_raw(features['image_raw'], tf.uint16)

#        image_shape = tf.pack([height, width, 1])
#        image = tf.reshape(image, image_shape)

#        # Perform additional preprocessing on the parsed data
#        return image

#    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
#    # tensor for each example.
#    dataset = dataset.map(parser)
#    dataset = dataset.shuffle(buffer_size=10000)
#    dataset = dataset.batch(32)
#    dataset = dataset.repeat(num_epochs)
#    iterator = dataset.make_one_shot_iterator()

#    # `features` is a dictionary in which each value is a batch of values for
#    # that feature; `labels` is a batch of labels.
#    features, labels = iterator.get_next()

#    return features, labels

def parser(record):
    '''Parse a TFRecord and return a training example'''
    features = {
        "image": tf.FixedLenFeature((), tf.string, ""),
    }
    parsed = tf.parse_single_example(record, features)
    img = tf.decode_raw(parsed["image"], tf.float32)

    image_shape = [height, width, 1]
    img = tf.reshape(img, image_shape)

    return img

def gen_lq(img, mean):
    '''Generate low quality image'''
    print(type(img))
    print(img)
    #Ensure that the seed is random
    np.random.seed(int(np.random.rand()*(2**32-1)))

    #Adjust the image scale so that the image has the correct average counts
    lq = np.random.poisson(mean * (img / np.mean(img)))
    
    #Rescale between 0 and 1
    min = np.min(lq)
    lq = (lq-min) / (np.max(lq)-min)

    return lq, img

def generator():
    yield gen_lq()

def input_fn(filenames, mean):
    """How records will be used"""

    files = tf.data.TFRecordDataset(filenames)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=parallel_readers)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(map_func=parser)
    dataset = dataset.map(lambda img: tuple(tf.py_func(gen_lq, [img, mean], [tf.float32, tf.float32])))
    #dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat(num_epochs)

    iter = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        next = iter.get_next()
        print(sess.run(next)) #Output

    lq, img = iter.get_next()

    return lq, img

def main(unused_argv):

    mean = 128 #Average value of pixels in low quality generated images
    
    lq_train, img_train = input_fn(tfrecords_train_filenames, mean)
    lq_val, img_val = input_fn(tfrecords_val_filenames, mean)
    lq_test, img_test = input_fn(tfrecords_test_filenames, mean)
    
    #Create the Estimator
    estimator = tf.estimator.Estimator(model_fn=cnn_model_fn_enhancer, model_dir=model_dir)

    # Set up logging for predictions. ADD METRICS LATER
    tensors_to_log = {  }#"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    #train_iter = train_data.make_one_shot_iterator()

    #next_example, next_label = iterator.get_next()

    with tf.Session() as sess:
        print(sess.run(lq_train)) # output

    #loss = loss_function(lq, img)

    #training_op = tf.train.AdagradOptimizer(...).minimize(loss)

    #with tf.train.MonitoredTrainingSession(...) as sess:
    #    while not sess.should_stop():
    #        sess.run(training_op)

    ##Batch normalisation
    ##update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    ##with tf.control_dependencies(update_ops):
    ##    #train_op = optimizer.minimize(loss)
    #with tf.Session() as sess:

    #    #Handle training data
    #    train_iter         = train_data.make_one_shot_iterator()
    #    train_iter_handle = sess.run(train_iter.string_handle())
    
    #    #Set up iteration
    #    handle = tf.placeholder(tf.string, shape=[])
    #    iterator = tf.data.Iterator.from_string_handle(
    #        handle, train_iter.output_types)
    #    lq, img = iterator.get_next()

    #    loss = loss_function(lq, img)

    #    training_op = tf.train.AdagradOptimizer(...).minimize(loss)
    #    train_loss = sess.run(loss, feed_dict={handle: train_iter_handle})

    #    print(train_loss)


    ##Generate data on the fly
    #train_data = train_data.map(lambda img: tuple(tf.py_func(
    #    gen_lq,
    #    [img, mean],
    #    [tf.float32, tf.float32])))

    ##Batch data
    #train_data.batch(batch_size)

    ##Batch normalisation
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    #    #train_op = optimizer.minimize(loss)
    #    with tf.Session() as sess:

    #        #Handle training data
    #        train_iter         = train_data.make_one_shot_iterator()
    #        train_iter_handle = sess.run(train_iter.string_handle())
    
    #        #Set up iteration
    #        handle = tf.placeholder(tf.string, shape=[])
    #        iterator = tf.data.Iterator.from_string_handle(
    #            handle, train_iter.output_types)
    #        next_element = iterator.get_next()
    
    #        #temp
    #        loss = next_element

    #        train_loss = sess.run(loss, feed_dict={handle: train_iter_handle})
    #        print(train_loss)

    #Classify training, validation and test data
    #train_data = tf.data.TFRecordDataset(tfrecords_train_filenames)
    #val_data = tf.data.TFRecordDataset(tfrecords_val_filenames)
    #test_data = tf.data.TFRecordDataset(tfrecords_test_filenames)

    #Process records to get training examples
    #train_data.map(map_func=parser)
    #val_data.map(map_func=parser)
    #test_data.map(map_func=parser)

    ##Generate data on the fly
    #train_data = train_data.map(lambda img: tuple(tf.py_func(
    #    gen_lq,
    #    [img, mean],
    #    [tf.float32, tf.float32])))
    #val_data = val_data.map(lambda img: tuple(tf.py_func(
    #    gen_lq,
    #    [img, mean],
    #    [tf.float32, tf.float32])))
    #test_data = test_data.map(lambda img: tuple(tf.py_func(
    #    gen_lq,
    #    [img, mean],
    #    [tf.float32, tf.float32])))

    ##Batch data
    #train_data.batch(batch_size)
    #val_data.batch(batch_size)
    #test_data.batch(batch_size)

    #train_iter         = train_data.make_initializable_iterator()
    #train_next_element = train_iter.get_next()

    #val_iter         = val_data.make_initializable_iterator()
    #val_next_element = val_iter.get_next()

    #test_iter         = test_data.make_initializable_iterator()
    #test_next_element = test_iter.get_next()

    ##Outer wrap important for batch normalisation to work
    #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #    with tf.Session() as sess:

    #        #Locate data
    #        feature = { 'image': tf.FixedLenFeature([], tf.string) }
    #        filename_queue = tf.train.string_input_producer(tfrecords_train_filenames, num_epochs=1)

    #        #Define a reader and read the next record
    #        reader = tf.TFRecordReader()
    #        _, serialized_example = reader.read(filename_queue)

    #        #Decode the record read by the reader
    #        features = tf.parse_single_example(serialized_example, features=feature)

    #        image = tf.decode_raw(features['image'], tf.float32)

    #        # Reshape image data into its original shape
    #        image_shape = [height, width, 1]
    #        image = tf.reshape(image, image_shape)

    #        #lq_np = gen_lq(image.eval(), mean)
    #        #lq_img = tf.constant(lq_np)

    #        # Creates batches by randomly shuffling tensors
    #        images = tf.train.shuffle_batch(
    #            [image], 
    #            batch_size=10, 
    #            capacity=64, 
    #            num_threads=1, 
    #            min_after_dequeue=10)

    #        # Initialize all global and local variables
    #        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #        sess.run(init_op)

    #        # Create a coordinator and run all QueueRunner objects
    #        coord = tf.train.Coordinator()
    #        threads = tf.train.start_queue_runners(coord=coord)

    #        #sess.run(train_iter.initializer)

    #        while True:
    #            try:
    #                img = sess.run([images])

    #                cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
    #                cv2.imshow("dfsd", img.reshape((2048,2048)))
    #                cv2.waitKey(0)

    #                #elem = sess.run(train_next_element)
    #                print('Success')
    #            except tf.errors.OutOfRangeError:
    #                print('End of dataset.')
    #                break

    #        # Stop the threads
    #        coord.request_stop()
    
    #        # Wait for threads to stop
    #        coord.join(threads)
    #        sess.close()