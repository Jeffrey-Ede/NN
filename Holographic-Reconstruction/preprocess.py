import numpy as np
import cv2

def preprocess(img, new_size):
    '''Prepare images to train the hologram reconstruction network with'''
    
    #Crop the image so that it is square
    h = img.shape[0]
    w = img.shape[1]
    img = img[0:h, 0:h] if h <= w else img[0:w, 0:w]

    #Resize the image to a smaller sqaure array
    img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_AREA)

    #Convert the image to float type and scale its values to lie between 0.0 and 1.0
    min = np.min(img)
    max = np.max(img)
    img = img.astype(np.float32)
    img = (img-min) / (max-min)

    return img

def apply_transforms(img):
    '''
    Apply a series of tranforms to an img and return a list containing them to increase the
    number of images in the dataset
    '''
    
    imgs = [img]

    #Horizontal and vertical flips
    imgs += [cv2.flip(img, 0), cv2.flip(img, 1)]

    #Rotate the img by 90, 180 and 270 deg
    imgs += [np.rot90(m)]
    imgs += [np.rot90(imgs[-1])]
    imgs += [np.rot90(imgs[-1])]

    #Flips and rotations
    imgs += [np.rot90(imgs[1]), np.rot90(imgs[2])]

    return imgs
