#name = "//flexo.ads.warwick.ac.uk/shared39/EOL2100/2100/Users/Jeffrey-Ede/Training-Data/1-100/test.tif"

import cv2
import numpy as np

def sigmoidalise(name):

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
    
    return fancy_sigmoidaliser(img)

#cv2.imshow("sdfsd", img)
#cv2.waitKey(0)

if __name__ == "__main__":

    import os

    dir = "/"
    new_dir = "/"

    #Sigmoidalise every file in the directory and save the sigmoidalisations
    for f in os.listdir(dir):

        sigmoidalise(dir+f).save(new_dir+f)
