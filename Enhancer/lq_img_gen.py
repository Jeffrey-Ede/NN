import time 
import numpy as np
import multiprocessing
import Image
import cv2

def lq_chunks_gen(rn_to_px_flat, chunksize):
    '''Generate one of the chunks to make low quality images from'''

    lq = np.zeros(rn_to_px_flat.shape, dtype=np.uint16)

    #Ensure that the random set is random
    rn = np.random.seed(int(np.random.rand()*(2**32-1)))
        
    #Generate large batch of random numbers
    rn = np.random.rand(chunksize)

    #Sort them into ascending order and add counts to the px they indicate
    sorted = np.sort(rn, axis=None)

    rn_idx = 0
    for x in sorted:
        if x < rn_to_px_flat[rn_idx]:
            lq[rn_idx] += 1
        else:
            rn_idx += 1
            lq[rn_idx] += 1

    return lq

def record_lq(lq, saveLoc, avg_counts):
    '''Rescale and save low quality image'''

    lq = 65535*(lq-np.min(lq)) / (np.max(lq)-np.min(lq))
    lq.astype(np.uint16)
    lq.reshape(img.shape)

    im = Image.fromarray(data)
    saveLoc = saveLoc if saveLoc[-1] is '/' else saveLoc+'/'
    cv2.imwrite(saveLoc+'counts'+str(avg_counts)+'.tif')

    return 

def lq_img_gen(img, saveLoc, avg_counts=[2, 4, 8, 16, 32, 64], chunkdepth = 2, num_parallel=2):
    '''Generate a low quality image from an unnormalised pdf'''
    avg_counts = np.sort(avg_counts)

    rn_to_px_flat = np.cumsum(img, axis=None)
    rn_to_px_flat /= rn_to_px_flat[-1]


    num_chunks = int(avg_counts[-1] / chunkdepth)

    #Add counts to low quality image in chunks
    chunksize = img.size * chunkdepth

    #lq = [np.zeros(rn_to_px_flat.shape,dtype=np.uint16)]*num_chunks
    
    #for i in range(num_chunks):

    #Prepare task parameters for chunk construction
    tasks = []
    for _ in range(num_chunks):
        tasks.append( (rn_to_px_flat, chunksize, ) )

    pool = multiprocessing.Pool(num_parallel)
    results = [pool.apply_async( lq_chunks_gen, t ) for t in tasks]
    chunks = [result.get() for result in results]

    #Sum together various numbers of chunks and prepare the results to be postprocessed
    #and saved in parallel
    chunk_sum_save_num = [avg / chunkdepth for avg in avg_counts]
    sum = np.zeros(chunks[0].shape)
    tasks = []
    savePos = 0
    for i, chunk in enumerate(chunks):
        sum += chunk
        if i == chunk_sum_save_num[savePos]:
            tasks.append( (sum, saveLoc, avg_counts[savePos], ) )
            savePos += 1

    pool = multiprocessing.Pool(num_parallel)
    results = [pool.apply_async( record_lq, t ) for t in tasks]
    results = [result.get() for result in results]

    return 

    #lq_chunks = pool.map(lq_chunks_gen, range(num_parallel))
    #pool.close()
    #pool.join()

    #for i in range(lq):
    #    lq[i] = 65535*(lq[i]-np.min(lq[i])) / (np.max(lq[i])-np.min(lq[i]))
    #    lq[i].astype(np.uint16)
    #    lq[i].reshape(img.shape)

    #return lq_chunks

if __name__ == "__main__":

    from scipy.misc import imread
    
    path = '//flexo.ads.warwick.ac.uk/shared39/EOL2100/2100/Users/Jeffrey-Ede/datasets/2100/'
    file = 'reaping10275.tif'
    
    img = imread(path+file, mode='F')
    #cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
    #cv2.imshow("dfsd", img.reshape((2048,2048)))
    #cv2.waitKey(0)
    #img = np.random.rand(2048*2048)
    print("hi"),
    tic = time.time()
    img = lq_img_gen(img, [2,4,8, 16], 2, 4)
    toc = time.time()
    print(toc-tic)
    print("hi2")
    cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
    cv2.imshow("dfsd", img[0].astype(np.float32).reshape((2048,2048)).normalize())
    cv2.waitKey(0)
    cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
    cv2.imshow("dfsd", img[1].astype(np.float32).reshape((2048,2048)))
    cv2.waitKey(0)
