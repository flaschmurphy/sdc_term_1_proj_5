import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import scipy.ndimage
import random
import sys

dstdir = 'training_images/my_images/car3/'
wsizes = [64, ]

def create_variant(img):
    if (random.choice([1, 0])):
        img = scipy.ndimage.interpolation.shift(img, [random.randrange(-3, 3), random.randrange(-3, 3), 0])
    else:
        img = scipy.ndimage.interpolation.rotate(img, random.randrange(-8, 8), reshape=False)
    return img


def random_translation(img):
    rows = img.shape[0]
    cols = img.shape[1]

    delta_x_max = cols//2
    delta_y_max = rows//2

    tr_x = np.random.uniform(-delta_x_max, delta_x_max)
    tr_y = np.random.uniform(-delta_y_max, delta_y_max)

    M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    img_out = cv2.warpAffine(img,M,(cols,rows))

    return img_out


get_fname = lambda: dstdir + str(np.random.randint(100000000, 999999999)) + '.jpg'
get_bright = lambda xy, wsize: tuple([_+wsize for _ in xy]) 
get_clr = lambda: np.random.randint(255, size=3).tolist() 
imsave = lambda img: cv2.imwrite(get_fname(), img)

fname = 'training_images/my_images/seed_car_2.png'
#fname = 'training_images/my_images/seed_not_car1.png'
#fname = 'training_images/my_images/seed2.png'
#fname = 'training_images/my_images/seed.png'
img = cv2.imread(fname)

#plt.ion()
#f = plt.figure(figsize=(20, 10))

xy = np.array((914, 424))
for wsize in wsizes:
    if wsize == 64:
        cnt = 0

        sys.stdout.write('\n')
        sys.stdout.flush()

        for y in range(0, 109, 3):
            for x in range(0, 141, 3):
                top_left = (xy[0]+x, xy[1]+y)
                bottom_right = (xy[0]+64+x, xy[1]+64+y)
                img_ = img.copy()
                img1 = img_[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], : ]


                #cv2.rectangle(img1, top_left, bottom_right, get_clr(), 2)
                #plt.clf()
                #plt.imshow(img_[:,:,::-1]) 
                #plt.show()
                #plt.pause(0.1)

                #img1[np.sum(img1, axis=2) > 550] = 0 
                imsave(img1)
                img2 = create_variant(img1)
                imsave(img2)
                img3 = random_translation(img1)
                imsave(img3)

                cnt += 1
                sys.stdout.flush()
                if cnt % 80 == 0:
                    sys.stdout.write('\n')
                sys.stdout.write('.')


sys.stdout.write('\n')

