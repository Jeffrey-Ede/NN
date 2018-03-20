import cv2
import numpy as np

import os

def sigmoidalise(name, size):

    def fancy_sigmoidaliser(img, scale=3): 
        '''
        Rescale intensities by projecting them with a signmoid based on their variance
        scale is the number of standard deviations to devide the image's deviations from the mean by
        '''
        mean = np.mean(img)
        sigma = np.sqrt(np.var(img))

        def sigmoidalise(x):

            res = (x-mean) / (scale*sigma)

            return np.exp(res) / (np.exp(res)+1.0)

        sigmoidaliser = np.vectorize(sigmoidalise, otypes=[np.float])

        return sigmoidaliser(img.copy())

    img = cv2.imread(name, cv2.IMREAD_UNCHANGED)

    #Crop image down to size, from the top left corner
    img = img[0:size, 0:size]
    
    return fancy_sigmoidaliser(img)

#cv2.imshow("sdfsd", img)
#cv2.waitKey(0)

if __name__ == "__main__":

    dir = "Documents/semantic_stacks/lacy_carbon/raw1/"
    new_dir = "Documents/semantic_stacks/lacy_carbon/train/"

    def multiple_mags(raw_dir, proc_dir, sizes, save_size):
        '''Sigmoidalise the images at multiple magnification scales'''

        for size in sizes:

            #Apply variance-based sigmoidalisation to reduce the effect of outliers of the network
            sigmoidalisation = sigmoidalise(raw_dir, size)

            #Resize the image to the specified save size
            if size is not save_size:
                sigmoidalisation = cv2.resize(img, (save_size, save_size),
                                             interpolation=cv2.INTER_AREA if size < save_size else cv2.INTER_CUBIC)

            sigmoidalisation.save(proc_dir)

    #Sigmoidalise every file in the directory and save the sigmoidalisations
    sizes = [668, 512, 384, 256, 192, 128]
    save_size = 256
    for f in os.listdir(dir):

        multiple_mags(dir+f, new_dir+f, sizes, save_size)
        
