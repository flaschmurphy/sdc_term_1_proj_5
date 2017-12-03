"""
 Udacity Self Driving Car Nanodegree

 Term 1, Project 5 -- Tests for Object Detection

 Author: Ciaran Murphy
 Date: 27th Nov 2017

"""
import pickle
import glob
import time
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec

from obj_detection import *

MODEL_FILE = 'model_05k.pkl'

def test_predictions():
    """Test a sequence of images on `predict()`."""
    print()
    print('Testing `predict()`')

    car_fnames = glob.glob('./test_images/car_*')
    notcar_fnames = glob.glob('./test_images/notcar_*')

    with open(MODEL_FILE, 'rb') as f:
        clf = pickle.load(f)

    for fname in car_fnames:
        img = imread(fname)
        predict(img, clf, fname=fname, car=True, vis=True)

    for fname in notcar_fnames:
        img = imread(fname)
        predict(img, clf, fname=fname, car=False, vis=True)

    plt.pause(2)
    plt.close()

    print()


def test_windowing():
    """Test boxes returned from `get_window_points()`. """
    print()
    #
    # Full image test
    #
    print('Testing `get_window_points()` on a full image...')
    img_size = (720, 1280, 3)
    window_size = (64, 64)
    overlap = 0.5
    start_pos = (0, 0)
    end_pos = (None, None)
    bboxes_full = get_window_points(img_size, window_size, overlap, start=start_pos, end=end_pos)
    assert len(bboxes_full) == 819, 'Incorrect number of windows, got {}, should be 819.'.format(
            len(bboxes_full))
    #
    # Partial image test - a band covering the bottom third of the frame
    #
    print('Testing `get_window_points()` on a partial image...')
    img_size = (720, 1280, 3)
    window_size = (64, 64)
    overlap = 0.5
    start_pos = (0, 360)
    end_pos = (1280, 490)
    bboxes_partial = get_window_points(img_size, window_size, overlap, start=start_pos, end=end_pos)
    assert len(bboxes_partial) == 117, 'Incorrect number of windows, got {}, should be 117.'.format(
            len(bboxes_partial))
    #
    # Partial image test - a band covering the bottom third of the frame, clipped at the left and right
    #
    print('Testing `get_window_points()` on a partial, clipped image...')
    img_size = (720, 1280, 3)
    window_size = (64, 64)
    overlap = 0.5
    start_pos = (256, 480)
    end_pos = (1024, 576)
    bboxes_clipped = get_window_points(img_size, window_size, overlap, start=start_pos, end=end_pos)

    assert len(bboxes_clipped) == 46, 'Incorrect number of windows, got {}, should be 46.'.format(
            len(bboxes_clipped))

    # Visualize the results
    vis = True
    if vis:
        fname = './test_images/bbox-example-image.jpg'
        get_clr = lambda: np.random.randint(255, size=3).tolist()

        plt.close()
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 10))

        canvas1 = imread(fname)
        for box in bboxes_full:
            cv2.rectangle(canvas1, box[0], box[1], get_clr(), 3)
        # Normalize for pyplot weirdness
        #canvas1 /= 255
        ax1.imshow(canvas1)

        canvas2 = imread(fname)
        for box in bboxes_partial:
            cv2.rectangle(canvas2, box[0], box[1], get_clr(), 3)
        # Normalize for pyplot weirdness
        #canvas2 /= 255
        ax2.imshow(canvas2)

        canvas3 = imread(fname)
        for box in bboxes_clipped:
            cv2.rectangle(canvas3, box[0], box[1], get_clr(), 3)
        # Normalize for pyplot weirdness
        #canvas3 /= 255
        ax3.imshow(canvas3)

        plt.tight_layout()
        plt.show()

        plt.pause(2)
        plt.close()

    print()


def test_sliding_window_predictions():
    """Test that a sliding window slides as expected and detects cars"""
    #
    # Run a scan over an image and retain any boxes where a car is detected
    #
    img = imread('./test_images/bbox-example-image.jpg')
    img_size = (720, 1280, 3)
    window_size = (64, 64)
    overlap = 0.5
    start_pos = (256, 510)
    end_pos = (1024, 576)
    bboxes = get_window_points(img_size, window_size, overlap, start=start_pos, end=end_pos)
    car_boxes = []

    # If True, show realtime visualizations
    vis = True    
    if vis:
        display_img = img.copy()    # Use this one for showing results
        plt.ion()    # Enable interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10)) 

    # Setup a CPU timer to find out how much time this all takes
    start = time.clock()

    # Load the classifier
    clf = get_svm_model(model_file=MODEL_FILE)

    get_clr = lambda: np.random.randint(255, size=3).tolist()

    for box in bboxes:
        top_left, bottom_right = box
        sub_img = img[ top_left[1]: bottom_right[1], top_left[0]: bottom_right[0], :]

        prediction = predict(sub_img, clf, fname='./test_images/bbox-example-image.jpg', vis=False, verbose=True)

        # Store any box that the classifier things contains a car
        if prediction == 1:
            car_boxes.append(box)

        if vis:
            display_img_ = display_img.copy()
            cv2.rectangle(display_img_, *box, get_clr(), 6)

            # If a car was predicted, keep the box painted
            if prediction == 1:
                display_img =  display_img_
            
            plt.sca(ax1)
            plt.cla()
            plt.sca(ax2)
            plt.cla()

            # The indexing and operation below converts BGR to RGB and normalizes to range 0,1
            #ax1.imshow(display_img_[:,:,::-1]/255)
            #ax2.imshow(sub_img[:,:,::-1]/255)
            imshow(display_img_, axis=ax1)
            imshow(sub_img, axis=ax2)

            plt.tight_layout()
            plt.show()
            plt.pause(0.0001)

    end = time.clock()
    duration = end - start

    plt.sca(ax1)
    plt.cla()
    ax1.imshow(display_img[:,:,::-1]/255)
    plt.show()
    plt.pause(10)

    print('Prcessing one image took {:.3}s in CPU time.'.format(duration))

        
def main():
    #test_predictions()
    #test_windowing()
    test_sliding_window_predictions()


if __name__ == '__main__':
    main()

    ###############################################################
    #
    # Test adding boundary boxes
    #
    ###############################################################
    #image = imread('./test_images/bbox-example-image.jpg')
    #bboxes = [((837, 507), (1120, 669))]
    #result = draw_boxes(image, bboxes)
    #imshow(result)()
    #plt.show()

    ###############################################################
    #
    # Muck around with boxing using cv2 image templates
    #
    ###############################################################
    #bboxs = template_matching()
    #image = imread('./test_images/bbox-example-image.jpg')
    #image = draw_boxes(image, bboxs)
    #imshow(image)()
    #plt.show()

    ###############################################################
    #
    # Show color histogram for color space
    #
    ###############################################################
    #img = imread('test_images/cutout1.jpg')
    #rhist, ghist, bhist, hist_features, bin_centers = color_hist(img, cspace='YCrCb', vis=True)
    #fig = plt.figure(figsize=(12,3))
    #plt.subplot(131)
    #plt.bar(bin_centers, rhist[0])
    #plt.xlim(0, 256)
    #plt.title('R Histogram')
    #plt.subplot(132)
    #plt.bar(bin_centers, ghist[0])
    #plt.xlim(0, 256)
    #plt.title('G Histogram')
    #plt.subplot(133)
    #plt.bar(bin_centers, bhist[0])
    #plt.xlim(0, 256)
    #plt.title('B Histogram')
    #plt.show()


    ###############################################################
    #
    # Explore RGB and HSV color spaces in 3D
    #
    ###############################################################
    ##files = glob.glob('./test_images/[0-9][0-9].png')
    #files = glob.glob('./test_images/*.png')
    #for f in files:
    #    print(f)
    #    img = imread(f)
    #    
    #    # Select a small fraction of pixels to plot by subsampling it
    #    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    #    img_small = cv2.resize(img, (np.int(img.shape[1] / scale),
    #        np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
    #    
    #    # Convert subsampled image to desired color space(s)
    #    img_small_RGB = img_small
    #    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
    #    img_small_HLS = cv2.cvtColor(img_small, cv2.COLOR_RGB2HLS)
    #    img_small_YCR = cv2.cvtColor(img_small, cv2.COLOR_RGB2YCrCb)
    #    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
    #    
    #    # Plot and show
    #    plot3d(img_small_RGB, img_small_rgb, axis_labels=list("RGB"))
    #    plt.show()
    #    
    #    plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    #    plt.show()

    #    plot3d(img_small_HLS, img_small_rgb, axis_labels=list("HLS"))
    #    plt.show()

    #    plot3d(img_small_YCR, img_small_rgb, axis_labels=list("YCrCb"))
    #    plt.show()


    ###############################################################
    #
    # Resizing images
    #
    ###############################################################

    #image = imread('test_images/cutout1.jpg')
    #feature_vec = bin_spatial(image, color_space='LUV', size=(32, 32))
    ## Plot features
    #plt.plot(feature_vec)
    #plt.title('Spatially Binned Features')
    #plt.show()
    ## Question - can we scale down even further? Consider this when training classifier

    ###############################################################
    #
    # Test the HOG function
    #
    ###############################################################

    # Generate a random index to look at a car image
    #d = load_data(car_or_not='car', random=True)
    #fname, image, idx = d[0]

    ## Convert to gray before sending to HOG
    #gray = rgb2gray(image)

    ## Define HOG parameters
    #orient = 9
    #pix_per_cell = 8
    #cell_per_block = 2

    ## Call our function with vis=True to see an image output
    #features, hog_image = get_hog_features(
    #        gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)

    ## Plot the examples
    #fig = plt.figure()
    #plt.subplot(121)
    #imshow(image, cmap='gray')()
    #plt.title('Example Car Image')
    #plt.subplot(122)
    #plt.imshow(hog_image, cmap='gray')
    #plt.title('HOG Visualization')
    #plt.show()


    ###############################################################
    #
    # Sanity check the data
    #
    ###############################################################
    #_ = load_data(sanity=True)

