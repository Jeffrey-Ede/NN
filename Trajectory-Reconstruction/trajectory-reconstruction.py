import numpy as np

import cv2
from scipy.misc import imread

#img = cv2.imread("C:/dump/test1.tif")
img = imread("//flexo.ads.warwick.ac.uk/Shared39/EOL2100/2100/Users/Jeffrey-Ede/datasets/stills/img1.tif", mode='F')

img = (img+1000)/5000
print(img)
cv2.imshow("sdfsd", img)
cv2.waitKey(0)

#Original rows and columns of images
rows0 = 2688
cols0 = 2672

rows = cols = 2048 #Size of images before resizing
size = 256 #Size of images to feed into the neural network

def gen_trajectory(probs=[0.1, 0.1, 0.1, 0.1], steps=64):
    '''Generate a trajectory for the sample to follow in the first quadrant'''

    #Get probabilites of various types of trajectories and use a
    #random number to decide which one to use
    cum_probs = [probs[0]]*len(gen_trajectory)
    for i in range(1, len(gen_trajectory)):
        cum_probs[i] = cum_probs[i-1] + probs[i]

    rand = np.random.random()
    type_num = -1
    for i, cum in enumerate(cum_probs):
        if cum <= rand:
            type_num = i
            break
    else:
        #In case something goes wrong...
        return -1

    #Generate a trajectory of the selected type

    traj = 1

    return traj

def imgs_from_traj(map, traj, steps):
    '''Generate images for motion along a given trajectory'''



    return 