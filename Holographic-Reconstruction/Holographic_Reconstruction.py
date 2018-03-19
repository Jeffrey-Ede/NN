from keras.layers import Input, AveragePooling2D, Conv2D, PReLU, add, Reshape, concatenate
from keras.models import Model

from keras import optimizers

#rows = 667
#cols = 672
rows = cols = 256 # For simplicity during development

channels = 2 #Amplitude and phase

feature_space_size = 32 #Number of features that the initial convolutional layer projects onto

#Input images
input = Input(shape=(rows, cols, channels), dtype='float32')
#input = AveragePooling2D(pool_size=(3,3))(input) #Downsample the input to decrease computation

#Downsample the image by various amounts to process features on various scales
d1 = AveragePooling2D(pool_size=(1,1))(input)
d2 = AveragePooling2D(pool_size=(2,2))(input)
d4 = AveragePooling2D(pool_size=(4,4))(input)
d8 = AveragePooling2D(pool_size=(8,8))(input)

def create_residual_block(input):
    '''Residually concatenates the block input after flowing it through 2 convolutional layers'''

    residual_block = Conv2D(filters=feature_space_size, kernel_size=(3,3), padding='same', activation='relu')(input)
    residual_block = Conv2D(filters=feature_space_size, kernel_size=(3,3), padding='same', activation='relu')(residual_block)
    residual_block = add([residual_block, input])

    return residual_block

def create_blockchain(input):
    '''Define the blockchain used for each of the downsamplings'''

    blockchain = Conv2D(filters=feature_space_size, padding='same', kernel_size=(3,3))(input) #No activation
    blockchain = create_residual_block(blockchain)
    blockchain = create_residual_block(blockchain)
    blockchain = create_residual_block(blockchain)
    blockchain = create_residual_block(blockchain)

    return blockchain

def create_upsampling_block(input, new_rows, new_cols):
    '''Use a convolutional layer to project the input into a larger feature space that can be reshaped to upsample the input'''

    upsampling_block = Conv2D(filters=4*feature_space_size, kernel_size=(3,3), padding='same', activation='relu')(input)
    upsampling_block = Reshape((int(new_rows), int(new_cols), feature_space_size))(upsampling_block)

    return upsampling_block

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
d1 = Conv2D(filters=feature_space_size, kernel_size=(3,3), padding='same')(d1)
d2 = Conv2D(filters=feature_space_size, kernel_size=(3,3), padding='same')(d2)
d4 = Conv2D(filters=feature_space_size, kernel_size=(3,3), padding='same')(d4)
d8 = Conv2D(filters=feature_space_size, kernel_size=(3,3), padding='same')(d8)

concatenation = concatenate([d1, d2, d4, d8])
concatenation = Conv2D(filters=2, kernel_size=(3,3), padding='same')(concatenation)

model = Model(inputs=input, outputs=concatenation)
model.summary()

##Compile the model for training
#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #ADAM optimisation
#model.compile(loss='mean_squared_error', optimizer=adam, =metrics=['accuracy'])

##Fit the model to the training data
#model.fit(shuffle=True)

##Save weights after training
#model.save('hologram_reconstructor.h5')