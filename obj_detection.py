"""
 Udacity Self Driving Car Nanodegree

 Term 1, Project 5 -- Object Detection

 Author: Ciaran Murphy
 Date: 27th Nov 2017

"""
import os
import glob
import cv2
import sys
import pickle
import tempfile
import shutil
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from moviepy.editor import VideoFileClip
from collections import deque

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.ndimage.measurements import label
from skimage.feature import hog

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D


CAR_FNAMES = glob.glob('training_images/my_images/car1/*')
CAR_FNAMES += glob.glob('training_images/my_images/car2/*')
CAR_FNAMES += glob.glob('training_images/my_images/car3/*')
CAR_FNAMES += glob.glob('./training_images/vehicles/*/*.png')
CAR_FNAMES += glob.glob('./training_images/vehicles/[!KITTI]*/*.png')

NOTCAR_FNAMES = glob.glob('training_images/my_images/notcar2/*')
NOTCAR_FNAMES += glob.glob('training_images/my_images/notcar3/*')
NOTCAR_FNAMES += glob.glob('training_images/my_images/notcar4/*')
NOTCAR_FNAMES += glob.glob('training_images/my_images/notcar1/*')
NOTCAR_FNAMES += glob.glob('./training_images/non-vehicles/[!KITTI]*/*.png')
NOTCAR_FNAMES += glob.glob('./training_images/non-vehicles/*/*.png')

NOTCAR_FNAMES = NOTCAR_FNAMES[:len(CAR_FNAMES)]

#######################################################################################################
#
# Helper functions and classes
#
#######################################################################################################

def parse_args():
    parser = ArgumentParser()

    ex_group = parser.add_mutually_exclusive_group(required=True)
    ex_group.add_argument('-t', '--train', dest='trainsize',
            help="""Include the training step and use TRAIN value as the number of _notcar_ samples to 
            use for training on. Specify 'all' to train on all available data. If not including 
            training, must specify a classifier to load from disk (a previously pickled one) 
            using the `-c` switch. The number of car samples loaded for training will be porportional
            to the number of notcar samples specified here.""")

    ex_group.add_argument('-v', '--videoin', dest='input_video',
            help="""The input video file""")

    parser.add_argument('-0', '--t0', dest='video_start',
            help="""T0 -- time in seconds to start the video from""")

    parser.add_argument('-1', '--t1', dest='video_end',
            help="""T1 -- time in seconds to end the video at""")

    parser.add_argument('-s', '--save', dest='save_file', 
            help="""Where to save the SVM model back to disk (if training)""")

    parser.add_argument('-o', '--videoout', dest='output_video',
            help="""The video output file""")
 
    parser.add_argument('-c', '--classifier', dest='clf_fname', 
            help="""Where to find a previously pickled SVM file to use if not training""")

    parser.add_argument('-d', '--stdscaler', dest='scaler_fname',
            help="""Where to find a previously pickled StandadrScaler for the SVM""")

    parser.add_argument('-g', '--debug', dest='debug', action='store_true',
            help="""Debug mode - things like include all rectangles to output video and 
            possibly print more logs.""")

    args = parser.parse_args()

    if args.trainsize is None:
        args.train = False
        if args.clf_fname is None:
            parser.print_usage()
            sys.exit()
        if args.scaler_fname is None:
            args.scaler_fname = ''.join(args.clf_fname.split('.')[:-1]) + '_scaler.pkl'

    else:
        args.train = True
        if args.trainsize == 'all':
            args.trainsize = -1
        else:
            args.trainsize = int(args.trainsize)
 
    if args.save_file is None:
        print('!!! WARNING !!! Trained model will not be stored to disk.')

    return args


def imread(fname, for_prediction=False):
    """Helper function to always load images in a consistent way. Note that since we're using cv2
    to load the images, they will be in BGR by default,"""
    img = cv2.imread(fname).astype(np.float32)
    if int(img.max()) <= 1:
        print('!!! WARNING !!! Image does not apper to be the right scale, found ({}, {}) for {}.'.format(
            img.min(), img.max(), fname))
    if for_prediction is True:
        img.resize((64, 64, 3))
    return img


def imshow(*args, **kwargs):
    """Helper function to always show images in a consistent way"""
    args_ = [i for i in args]
    imgc = args_[0].copy()

    # Because I'm using cv2 to load the images, they are in BGR format
    # and scalled to 0,255. Fix this before displaying in pyplot as follows
    imgc = imgc[:,:,::-1]/255
    args_[0] = imgc

    if 'axis' in kwargs:
        axis = kwargs.pop('axis')
        axis.imshow(*args_, **kwargs)
    else:
        plt.imshow(*args_, **kwargs)


def imsave(fname, img):
    """Save an image to disk in a consistent way"""
    cv2.imwrite(fname, img)


def color_xform(img, tcmap):
    """Convert img from BGR to the target color space

    Args:
        img: input imge in BGR, range 0:255
        tcmap: target color space (RGB, HSV, LUV, HLS, YUV, YCrCb)

    Returns:
        a new image in the target color space

    """
    assert tcmap in ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'], 'Invalid target color space'

    img = img.copy()
    if tcmap == 'BGR': 
        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif tcmap == 'HSV':
        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif tcmap == 'LUV':
        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif tcmap == 'HLS':
        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif tcmap == 'YUV':
        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif tcmap == 'YCrCb':
        img_new = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    else: 
        img_new = img
    assert img_new.max() > 1, "Pixel value range is not (0, 255), it's {}".format((img.min(), img.max()))

    return img_new


def draw_labeled_bboxes(img, labels):
    """Take in a labels canvas and use it to overlay bounding rectangles for detected objects.
    
    Args:
        img: the image to modify
        labels: at 2-tuple containing the box image and the number of objects

    Returns:
        a modified verion of the image with boxes added (changes in place)

    """
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Define a bounding box based on min/max x and y
        x1, y1, x2, y2 = np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)
        min_side_length = 64
        if x2 - x1 < min_side_length:
            mid = x1 + ((x2 - x1) // 2)
            x1 = x1 + mid - (min_side_length//2)
            x2 = x1 + min_side_length

        if y2 - y1 < min_side_length:
            mid = y1 + ((y2 - y1) // 2)
            y1 = y1 + mid - (min_side_length//2)
            y2 = y1 + min_side_length

        bbox = ((x1, y1), (x2, y2))

        # Draw the box on the image
        img = cv2.rectangle(img, bbox[0], bbox[1], (11,102,0), 4)

        return img


def video_extract(src_fname, t1, t2, tgt_fname=None):
    """Extract a subset of a video and save it to disk. The ffmpeg_tools way is pretty fast.
    
    Args:
        src_fname: source video
        t1: start time in seconds
        t2: end time in seconds
        tgt_fname: where to store the clip. If None a temp file will be created and it's path returned
    
    """
    if tgt_fname is None:
        tgt_fname = tempfile.mkstemp(suffix='.mp4')[1]
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    ffmpeg_extract_subclip(src_fname, t1, t2, targetname=tgt_fname)

    return tgt_fname


class Bunch():
    """Convience class to enable dot style notation on a dict"""
    def __init__(self, adict):
        self.__dict__.update(adict)
    def __repr__(self):
        return self.__dict__.__repr__()


#######################################################################################################
#
# Pipeline functions
#
#######################################################################################################

def get_template_matches(img, template_fnames, method=cv2.TM_CCOEFF_NORMED):
    """Define a function to search for template matches and return a list of
    bounding boxes. Code is from lesson 9 of the project. This method should
    not be used as it is not portable. Template matching will only work when
    the image being scanned contains the template object in the same color,
    scale and orientation. Use template matching for objects that don't vary
    in apperance, not for real world objects.

    Args: 
        img: the image to search through
        template_fnames: list of image templates file names (they are read 
            from disk in this method)
        method: cv2.mathTemplate matching method. Other options include: 
            cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCORR, cv2.TM_SQDIFF,
            cv2.TM_SQDIFF_NORMED

    Returns:
        list of bounding boxes that were found

    """
    bbox_list = []
    for template in template_fnames:
        tmp = imread(template)
        result = cv2.matchTemplate(img, tmp, method)

        # Extract the location of the best match. `min_val` and `max_val` are not 
        # used in reality
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Determine a bounding box for the match
        w, h = (tmp.shape[1], tmp.shape[0])

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        bbox_list.append((top_left, bottom_right))
        
    return bbox_list


def get_colorhist(img, nbins=64, tcmap='HLS', bins_range=(0, 256), vis=False):
    """Generate a histogram of color channels (RGB).

    Args:
        img: the imput image
        nbins: the number of bins in the histogram
        tcmap: target color map. the image will be converted to this color space 
            before processing
        bins_range: the range for the bins
        vis: if True, output data for creating a visualization

    Returns:
        red histogram
        green histogram 
        blue histogram 
        hist_features (contatenated r,g,b histograms)
        bin_centers

    """
    assert img.max() > 1, "Pixel value range is not (0, 255), it's {}".format((img.min(), img.max()))

    img = color_xform(img, tcmap)
    ch1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ch2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    ch3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))

    if vis is False:
        return hist_features

    bin_edges = ch1_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2

    assert len(hist_features) > 0, 'Got no color historgram for image'

    return ch1_hist, ch2_hist, ch3_hist, hist_features, bin_centers


def get_spatial(img, tcmap='LUV', size=(16, 16)):
    """Convert an image to the provided color space, rescale it and unroll it.

    Args:
        img: the input image (assumed to be in RGB color space)
        tcmap: target color map. the image will be converted to this color space 
            before processing
        size: target size to scale to

    Returns:
        one dimensional feature vector of the converted image

    """
    img = color_xform(img, tcmap)
    assert img.max() > 1, "Pixel value range is not (0, 255), it's {}".format((img.min(), img.max()))
    features = cv2.resize(img, size).ravel()
    assert len(features) > 0, 'Got no spatial features for image'

    return features


def get_hog(img, orient=12, pix_per_cell=16, cell_per_block=2, channel='all', tcmap='LUV'):
    """Get a Histogram of Oriented Gradients for an image.

    "Note: you could also include a keyword to set the tranform_sqrt flag but
    for this exercise you can just leave this at the default value of
    transform_sqrt=False"

    Args:
        img: the input image
        orient: the number of orientation bins in the output histogram. Typical values are between 6 and 12
        pix_per_cell: a 2-tuple giving the number of pixels per cell e.g. `(9, 9)`
        cells_per_block: a 2-tuple giving the number of cells per block
        channel: which of the 3 color channels to get the HOG for, or 'all'
        tcmap: target color map. the image will be converted to this color space 
            before processing

    Returns:
        (features, hog_image) were the features are rolled out as a 1-D vector

    """
    assert channel in [0, 1, 2, 'all'], "Invalid channel specified, must be one of [0, 1, 2 'all']"
    img = color_xform(img, tcmap)
    if channel == 'all':
        all_features = []
        for i in (0, 1, 2):
            features, hog_image = hog(
                    img[:,:,i], 
                    orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell), 
                    cells_per_block=(cell_per_block, cell_per_block), 
                    transform_sqrt=False,
                    visualise=True, 
                    feature_vector=True,
                    block_norm="L2-Hys"
                )
            all_features.append(features)
        features = np.hstack(all_features).ravel()
    else:
        features, hog_image = hog(
                img[:,:,channel], 
                orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell), 
                cells_per_block=(cell_per_block, cell_per_block), 
                transform_sqrt=False,
                visualise=True, 
                feature_vector=True,
                block_norm="L2-Hys"
            )
        features = features.ravel()

    return features


def pipeline(img_data):
    """Extract features to generate a feature vector.

    Args:
        img_data: a list of 3-tuples containing (filename, RGB image, idx) where idx is the index of 
            where that image was found in the original training data
        spacial_size: input to be used for the `bin_spacial()` function
        hist_bins: number of bins for the `get_colorhist()` function
        hist_range: hist range to pass to the `get_colorhist()` function
        vis: if True, plot the results in a window and also store these visualizations
            to a local folder called './resources' for offline viewing

    Returns:
        single list of normalized features

    """
    features = []

    ctr = 0
    for fname, img, idx in img_data:
        assert img.max() > 1, "Pixel value range is not (0, 255), it's {}".format((img.min(), img.max()))

        spatial_features = get_spatial(img)
        color_features = get_colorhist(img)
        hog_features = get_hog(img)
        features.append(np.concatenate((
                spatial_features, 
                color_features, 
                hog_features,
           )).astype(np.float32))
        ctr += 1

    assert len(features) > 0, 'Got no features'

    return features


def video_pipeline(img):
    """Callback for moviepy video processing.

    Args:
        img: the input image, coming from moviepy will be in RGB

    Returns:
        a new image with boxes drawn where cars are predicted

    """
    global args, conf
    
    if 'cnt' not in video_pipeline.__dict__:
        video_pipeline.cnt = 0
    video_pipeline.cnt += 1

    if video_pipeline.cnt < 145:
        conf.img_hist.append(img)
        return img

    if args.debug and video_pipeline.cnt % conf.n != 0:
        return conf.img_hist[-1]

    fname = 'video_frame_{:04}.png'.format(video_pipeline.cnt) # a dummy file name needed by pipeline()

    # Convert from RGB which is what moviepy loads in to BGR which is what the
    # rest of this script us using (cv2 default)
    img = img.copy()[:,:,::-1] 

    # Lambda to generate a random color
    get_clr = lambda: np.random.randint(255, size=3).tolist()


    ###############################################################################################################
    #
    # Define a method to sanity check the boxes and append them to the main array if they look ok
    #
    ###############################################################################################################
    frame_boxes = []
    to_be_painted = []

    def process_boxes(pstr, img, img_dbg):
        for box in bboxes:
            top_left, bottom_right = box
            sub_img = img[ top_left[1]: bottom_right[1], top_left[0]: bottom_right[0], :]
            prediction = predict(sub_img, conf.clf, args.scaler_fname, fname=pstr + fname, vis=False, verbose=True)
            if prediction == 1:
                frame_boxes.append(box)
                img, img_dbg = process_box(img, img_dbg, box, to_be_painted)
        return img, img_dbg

    def process_box(img, img_dbg, box, to_be_painted):
        if conf.enable_boxhist is False:
            to_be_painted.append(box)
            return img, img_dbg

        # Find the center of the new box
        center = (box[0][0] + ((box[1][0] - box[0][0]) // 2), box[0][1] + ((box[1][1] - box[0][1]) // 2))

        # If the new box has it's center contained within the boxes discovered
        # in the last N frames (= hist_size) consider it valid otherwise reject
        # it. If in debug mode, draw a circle to indicate that it was rejected.
        hits = 0
        for pastframe in conf.frame_box_history:
            for pastbox in pastframe:
                if (center[0] > pastbox[0][0] and center[0] < pastbox[1][0] 
                    and center[1] > pastbox[0][1] and center[1] < pastbox[1][1]):
                        hits += 1
                        break

        if hits == conf.hist_size and video_pipeline.cnt > conf.hist_size:
            if args.debug:
                img_dbg = cv2.circle(img_dbg, center, 20, (0,200,200), 1)
            to_be_painted.append(box)
        elif args.debug:
            img_dbg = cv2.circle(img_dbg, center, 20, (0,0,200), 1)

        return img, img_dbg

    ###############################################################################################################

    img_dbg = img.copy()

    #
    # Far bounding boxes
    #
    bboxes = get_window_points(img.shape, conf.far_window_size, conf.far_overlap, 
            start=conf.far_start_pos, end=conf.far_end_pos)
    #_boxes = bboxes.copy()
    img, img_dbg = process_boxes('far_', img, img_dbg)
    
    #
    # Middle bounding boxes
    #
    bboxes = get_window_points(img.shape, conf.mid_window_size, conf.mid_overlap, 
            start=conf.mid_start_pos, end=conf.mid_end_pos)
    img, img_dbg = process_boxes('mid_', img, img_dbg)
     
    #
    # Near bounding boxes
    #
    bboxes = get_window_points(img.shape, conf.near_window_size, conf.near_overlap, 
            start=conf.near_start_pos, end=conf.near_end_pos)
    img, img_dbg = process_boxes('near_', img, img_dbg)

    # Add the car predictions for this frame to the history
    conf.frame_box_history.append(frame_boxes)

    # Some annoying bug in cv2 that needs this copy workaround :( :(
    img = img.copy()
    img_dbg = img_dbg.copy()

    # For visualizing the process, draw boxes on a debug image
    if args.debug is True:
        # Draw all candidate boxes
        #for box in _boxes:
        #    img_dbg = cv2.rectangle(img_dbg, box[0], box[1], get_clr(), 1)
        # Draw all positive prediction boxes before any filtering (white)
        for box in frame_boxes:
            img_dbg = cv2.rectangle(img_dbg, box[0], box[1], (255, 255, 255), 1)
        # Draw all positive prediction boxes after the history filter (yellow)
        for box in to_be_painted:
            img_dbg = cv2.rectangle(img_dbg, box[0], box[1], (0, 255, 255), 1)

    # Increment the heatmap, conpare it to history and draw the labels
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    for box in to_be_painted:
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    if args.debug:
        plt.imsave(conf.dst1 + '{:04}.jpg'.format(video_pipeline.cnt), heatmap, cmap='gray')

    heatmap[heatmap < conf.threshold] = 0

    if args.debug:
        plt.imsave(conf.dst2 + '{:04}.jpg'.format(video_pipeline.cnt), heatmap, cmap='gray')

    if conf.enable_heathist:
        conf.frame_heat_history.append(heatmap)
        labels = label(np.sum(conf.frame_heat_history, axis=0))
    else:
        labels = label(heatmap)

    if args.debug:
        plt.imsave(conf.dst3 + '{:04}.jpg'.format(video_pipeline.cnt), heatmap, cmap='gray')

    if labels[1] > 0 :
        img = draw_labeled_bboxes(img, labels)
        img_dbg = draw_labeled_bboxes(img_dbg, labels)

    # Convert back to RGB for moviepy
    img = img[:,:,::-1]
    img_dbg = img_dbg[:,:,::-1]

    if args.debug is True:
        conf.img_hist.append(img_dbg)
        return img_dbg
    else:
        conf.img_hist.append(img)
        return img


#######################################################################################################
#
# Data model functions
#
#######################################################################################################

def load_data(car_or_not='car', ratio=1, length=-1, random=False, sanity=True, vis=False):
    """Load all images from disk. Loads using matplotlib --> range will be 0 to 1.

    Args:
        car_or_not: if 'car', return a generator of car images, if 'not', return a generator 
            for images that are not of cars
        ratio: the ration of car to noncar images to load. E.g. 0.5 means load half as many
            car images as notcar. This has the effect of biasing the model towards notcar 
            detection which makes sense because most of the visual space in the images does 
            not contain a car.
        length: the amount of data to load per car or notcar, meaning a value of 2500 will
            try to load 2500 car or notcar images depending on the value of car_or_not
        random: if True, instead of returning many images, return a single fname,
            image and index selected at random
        sanity: if True print out some info about the data and verify it's consistency
        vis: if True, display a random sample image

    Returns:
       a list of 2-tuples containing (filename, image). Note: the color space is RGB.

    """
    assert car_or_not in ['car', 'notcar'], "Invalid option for 'car_or_not'. Chose from ['car', 'notcar']"

    car_fnames = CAR_FNAMES
    notcar_fnames = NOTCAR_FNAMES

    # Verify that all images are of the same shape and dtype
    if sanity:
        print('Checking that all images have the same size, dtype and range...')

        first_img = imread(car_fnames[0])
        image_shape = first_img.shape
        data_type = first_img.dtype
        file_types = defaultdict(lambda: 0)

        def get_minmax(img):
            if img.min() <= 1 and img.min() >= 0: 
                minval = 0
            elif img.min() < 0:
                minval = img.min()
            else:
                minval = 0

            if img.max() > 1 and img.max() <= 255: 
                maxval = 255
            elif img.max() <= 1:
                maxval =  1
            else:
                maxval = img.max()
            return minval, maxval

        first_minval, first_maxval = get_minmax(first_img)

        for fname in car_fnames + notcar_fnames:

            ext = fname.split('.')[-1]
            file_types[ext] += 1

            cur_img = imread(fname)

            curmin, curmax = get_minmax(cur_img)
            if cur_img.shape != image_shape:
               raise Exception('Not all images have the same shape. {} was of shape {}.'.format(fname, cur_img.shape))
            if cur_img.dtype != data_type:
               raise Exception('Not all images have the same data type. {} was of type {}.'.format(fname, cur_img.dtype))
            if curmin != first_minval or curmax != first_maxval:
                raise Exception(
                        """Not all images have the same dinemsions, got: ({}, {}) for {}, but expected ({}, {})"""\
                                 .format(curmin, curmax, fname, first_minval, first_maxval))
        print('  All images are consistent!')
 
        data = {
               'n_notcars': len(notcar_fnames), 
               'n_cars': len(car_fnames),
               'image_shape': image_shape,
               'data_type': data_type,
               }

        print('  Total number of images: {} cars and {} non-cars'.format(data["n_cars"], data["n_notcars"]))
        print('  Image size: {}, data type: {}'.format(data["image_shape"], data["data_type"]))
        print('  Pixel value range: ({}, {})'.format(first_minval, first_maxval))
        print('  File type counts: {}'.format(str(dict(file_types))))

    if car_or_not == 'car':
        if length < 0:
            car_length = len(car_fnames)
        else:
            car_length = length
        fnames = np.random.permutation(car_fnames).tolist()[:int(car_length*ratio)]
    else:
        if length < 0:
            notcar_length = len(notcar_fnames)
        else:
            notcar_length = length
        fnames = np.random.permutation(notcar_fnames).tolist()[:notcar_length]

    ret = []
    if random is True:
        idx = np.random.randint(0, len(fnames))
        img = imread(fnames[idx])

        # The classifier seems to have a problem with the white car in the 
        # video so I'm changing it to a black car using the line below. 
        #img[np.sum(img, axis=2) > 550] = 0

        ret.append([fnames[idx], img, idx])
    else:
        i = 0
        for f in fnames:
            img = imread(f)

            # The classifier seems to have a problem with the white car in the 
            # video so I'm changing it to a black car using the line below. 
            #img[np.sum(img, axis=2) > 550] = 0

            ret.append([f, img, i])
            i += 1

    return ret


def get_xy(clf_fname=None, length=-1):
    """Load image data and extract features and labels for training

    Args:
        clf_fname: the file name where the classifier will be stored. This is used to generate
            a filename for the StandardScaler object to also be stored.
        length: the amount of data to load per car or notcar, meaning a value of 2500 will
            try to load 2500 car or notcar images depending on the value of car_or_not

    Returns:
        X matrix of features and y vector of labels.

    """

    print('Loading features and labels...')

    car_data = load_data(length=length, car_or_not='car')
    notcar_data = load_data(length=length, car_or_not='notcar', sanity=False)

    car_features = pipeline(car_data)
    notcar_features = pipeline(notcar_data)

    # Create an array stack of feature vectors
    X_not_scaled = np.vstack((car_features, notcar_features)).astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_not_scaled)

    # Very important that the StandardScaler that's made here is reused for all future predictions
    if clf_fname is not None:
        scaler_fname = ''.join(clf_fname.split('.')[:-1]) + '_scaler.pkl'
        pickle.dump(scaler, open(scaler_fname, 'wb'))

    # Generate the labels
    car_labels = np.ones(len(car_features))
    notcar_labels = np.zeros(len(notcar_features))
    y = np.append(car_labels, notcar_labels)

    print('Loaded, extracted features, scaled and labeled {:,} images, {:,} cars and {:,} not cars'.format(
        len(y), len(car_data), len(notcar_data)))
    print('StandardScaler was pickled to {}'.format(scaler_fname))

    return X, y


def get_svm_model(train=False, X=None, y=None, subset_size=None, model_file=None):
    """Create a Support Vector Machine and GridSearchCV for optimal params.

    Args:
        train: if True, training will be performed, otherwise a model is loaded 
            from disk
        X: the feature data
        y: the label data
        subset_size: if not none, train on a subset of data this size, otherwise
            train on all data
        model_file: if not training, load a pickled model from disk at this location
            and return that instead

    Returns:
        a trained SVM model

    """
    global conf

    if train is False:
        assert model_file is not None, 'Must specify a model file if not training'
        assert os.path.exists(model_file), 'Model file does not exist'

        with open(model_file, 'rb') as f:
            clf = pickle.load(f)

        print('Model was loaded from disk at location: {}.'.format(model_file))
        [print('  ' + l) for l in clf.__repr__().split('\n')]
        return clf

    assert X is not None, 'Must supply features for training'
    assert y is not None, 'Must supply labels for training'
    assert X.shape[0] == y.shape[0], 'Features and labels must be the same length'

    # Set the amount of data to use for training the classifier.
    if subset_size is None or subset_size < 0:
        subset_size = len(y)

    idx = np.random.permutation(X.shape[0])[:subset_size]
    X_subset = X[idx]
    y_subset = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=42, shuffle=True)

    print('Fitting data...')

    # Parameters for GridSearchCV
    grid_search = False
    parameters = [
            {'kernel': ['linear'], 'C': [0.001, 0.01, 1, 10]},
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.001, 0.001, 1, 10, 100]},
        ]
    if grid_search:
        clf = GridSearchCV(SVC(probability=True), parameters, cv=3, verbose=9)
    else:
        clf = SVC(kernel='linear', C=1, probability=True) 

    clf.fit(X_train, y_train)
    print(clf)

    if grid_search:
        print('Best Params: {}'.format(clf.best_params_))
        print('Best Score: {}'.format(clf.best_score_))
    print('Final score on test set: {}'.format(clf.score(X_test, y_test)))

    print()

    return clf


def predict(img, clf, scaler_pkl, fname='unknown', car=None, vis=False, verbose=False):
    """Extract features and run the classifier on an image.
    
    Args:
        img: the image to predict against
        clf: the classifier
        scaler_pkl: the pickled scaler for this model (must be reused from when the data
            was originally fit)
        fname: the filename of the original file containing the image
        car: if known (while testing), whether or not this is a car image
        vis: if True, display a visualization of the pipeline
        verbose: if True, print stuff
        
    Returns:
        The prediction is 1 if the prediction is 'car' or 0 if the prediction is 'not car'
        
    """
    img_new = cv2.resize(img, (64, 64))
    features = pipeline([[fname, img_new, 0]])

    if 'scaler' not in predict.__dict__:
        scaler = pickle.load(open(scaler_pkl, 'rb'))
        predict.scaler = scaler
    else:
        scaler = predict.scaler

    features = scaler.transform(features)
    prediction = clf.predict(features)[0]
    probs = clf.predict_proba(features)[0]

    if car is not None or verbose:    # meaning, someone told this method the truth value, so print stuff
        prediction_txt = 'car'
        if prediction == 0:
            prediction_txt = 'notcar'

        if car is None:
            grade = '??????'
        else:
            grade = 'CORRECT'
            if prediction == 0 and car:
                grade = 'INCORRECT'
            if prediction == 1 and not car:
                grade = 'INCORRECT'

        if prediction == 1:
            print('{:35}, predicts: {} = {:6} --> {:9}, probabilities: ({}, {})'.format(
                fname, prediction, prediction_txt, grade, probs[0], probs[1]))

    return int(prediction)


#######################################################################################################
#
# Main function
#
#######################################################################################################

def main():
    """Main entry point 
    """
    global conf

    # Create a history deque to track car detection boxes
    main.hist_size = 5
    main.enable_boxhist = True

    # Also create a history of heatmaps. The sum of all heatmaps will be 
    # used to detect objects instead of using just the heatmap from the current
    # image
    main.heat_size = 5
    main.enable_heathist = True

    # Threshold for the heatmap. Any pixels in the heatmap less than this 
    # value will be forced to zero
    main.threshold = 3

    # To improve processing times, only process every Nth frame when debugging.
    # For all other frames, return the current image with the last known rectangles redrawn
    main.n = 2

    #
    # Far bounding boxes
    #
    main.far_window_size = (64, 64)
    main.far_overlap = 0.25
    main.far_start_pos = (700, 400)
    main.far_end_pos = (1200, 685)
    #
    # Middle bounding boxes
    #
    main.mid_window_size = (128, 128)
    main.mid_overlap = 0.3
    main.mid_start_pos = (700, 380)
    main.mid_end_pos = (1280, 580)
    #
    # Near bounding boxes
    #
    main.near_window_size = (192, 192)
    main.near_overlap = 0.5
    main.near_start_pos = (700, 375)
    main.near_end_pos = (1280, 685)
    #

    main.frame_box_history = deque(maxlen=main.hist_size)
    main.frame_heat_history = deque(maxlen=main.heat_size)
    main.img_hist = deque(maxlen=main.n)

    if args.train is False:
        main.clf = get_svm_model(model_file=args.clf_fname)

        # The lines below create dirs that will be used to store the
        # heat map pipeliine in static images. This is very helpful 
        # for debugging. Note that everytime this script is called
        # these dirs are wiped and recreated from scratch.
        if args.debug:
            main.dst1 = './output/heat1/'
            main.dst2 = './output/heat2/'
            main.dst3 = './output/heat3/'
            if os.path.exists(main.dst1):
                shutil.rmtree(main.dst1)
            if os.path.exists(main.dst2):
                shutil.rmtree(main.dst2)
            if os.path.exists(main.dst3):
                shutil.rmtree(main.dst3)
            os.makedirs(main.dst1)
            os.makedirs(main.dst2)
            os.makedirs(main.dst3)

        if args.input_video is not None:
            assert args.output_video is not None, 'Must specify an output video.'
            assert args.clf_fname is not None, 'Must provide an SVM model.'
            
            print('Processing video...')
            if args.video_start is not None:
                assert args.video_end is not None, 'Must specify an end time as well'
                input_video = video_extract(args.input_video, int(args.video_start), int(args.video_end))
            else:
                input_video = args.input_video

            vin = VideoFileClip(input_video)
            conf = Bunch(main.__dict__)
            vout = vin.fl_image(video_pipeline)
            vout.write_videofile(args.output_video, audio=False)

    else:
        if args.save_file is not None and os.path.exists(args.save_file): 
            print('!!! WARNING !!! Previous model ({}) to be overwritten.'.format(args.save_file))

        X, y =  get_xy(clf_fname=args.save_file, length=args.trainsize)
        clf = get_svm_model(train=True, X=X, y=y, subset_size=args.trainsize, model_file=args.clf_fname)

        if args.save_file is not None:
            pickle.dump(clf, open(args.save_file, 'wb'))
            print('Model was saved to {}'.format(args.save_file))


#######################################################################################################
#
# Other functions
#
#######################################################################################################

def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D.

    Args:
        pixels: the input image
        colors_rgb: image in RGB scaled to [0,1] for plotting
        axis_labels: labels for the axis
        axis_limits: axis limits

    Returns:
       Axes3D object for further manipulation

    """

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')

    return ax


def get_window_points(img_size, window_size=(64, 64), overlap=0.5, start=(0, 0), end=(None, None)):
    """Generate top left and bottom right rectangle corners for boundary boxes.

    Args:
        img_size: the `img.shape()` of the image/canvas to generate the boxes for, e.g. (960, 1280, 3)
        window_size: the size of the sliding window in (x, y) order
        overlap: by how much should the windows overlap, e.g. 50% would be 0.5
        start: the (x, y) coordinate to start from
        end: the (x, y) coordinate in the image to stop at

    Returns:
        a list of (top_left, bottom_right) tuples that can be passed to `cv2.rectangle()`

    """

    size_x, size_y = window_size
    x_positions = []
    y_positions = []

    if end == (None, None):
        end = (img_size[1], img_size[0])

    start_x = start[0]
    end_x = end[0]
    while start_x <= (end_x - size_x):
        x_positions.append((start_x, start_x+size_x))
        start_x += int(size_x*overlap)

    start_y = start[1]
    end_y = end[1]
    while start_y <= (end_y - size_y):
        y_positions.append((start_y, start_y+size_y))
        start_y += int(size_y*overlap)

    bboxes = []
    for y in range(len(y_positions)):
        for x in range(len(x_positions)):
            bboxes.append([_ for _ in zip(x_positions[x], y_positions[y])])

    return bboxes


if __name__ == '__main__':
    global args
    args = parse_args()
    main()


