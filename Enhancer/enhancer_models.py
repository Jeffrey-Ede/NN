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