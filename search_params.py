import os
import glob
import cv2
import sys
import pickle
import tempfile
import shutil
import numpy as np
from collections import defaultdict

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

import pandas as pd
from io import StringIO

CAR_FNAMES = glob.glob('training_images/my_images/car2/*.jpg')[:2000]
CAR_FNAMES += glob.glob('training_images/my_images/car3/*.jpg')[:2000]
CAR_FNAMES += glob.glob('./training_images/vehicles/*/*.png')
NOTCAR_FNAMES = glob.glob('training_images/my_images/notcar/*.png')[:3000]
NOTCAR_FNAMES += glob.glob('training_images/my_images/notcar2/*.jpg')[:3000]
NOTCAR_FNAMES += glob.glob('./training_images/non-vehicles/*/*.png')
NOTCAR_FNAMES = NOTCAR_FNAMES[:len(CAR_FNAMES)]

idx = np.random.permutation(len(CAR_FNAMES))
n = 10
CAR_FNAMES = np.array(CAR_FNAMES)[idx].tolist()[:n]
NOTCAR_FNAMES = np.array(NOTCAR_FNAMES)[idx].tolist()[:n]


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
    assert tcmap in ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb', 'GRAY'], 'Invalid target color space'

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
    elif tcmap == 'GRAY':
        img_new = np.reshape(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (img.shape[0], img.shape[1], 1))
    else: 
        img_new = img
    assert img_new.max() > 1, "Pixel value range is not (0, 255), it's {}".format((img.min(), img.max()))

    return img_new


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


def get_hog(img, orient=10, pix_per_cell=16, cell_per_block=2, channel='all', tcmap='HSV', xform=False):
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
        xform: Boolean indicating whether or not to enable transform_sqrt

    Returns:
        (features, hog_image) were the features are rolled out as a 1-D vector

    """
    assert channel in [0, 1, 2, 'all'], "Invalid channel specified, must be one of [0, 1, 2 'all']"
    img = color_xform(img, tcmap)
    if channel == 'all':
        all_features = []
        for i in (range(img.shape[2])):
            features, hog_image = hog(
                    img[:,:,i], 
                    orientations=orient,
                    pixels_per_cell=(pix_per_cell, pix_per_cell), 
                    cells_per_block=(cell_per_block, cell_per_block), 
                    transform_sqrt=xform,
                    visualise=True, 
                    feature_vector=True,
                    block_norm="L2-Hys",
                )
            all_features.append(features)
        features = np.hstack(all_features).ravel()
    else:
        features, hog_image = hog(
                img[:,:,channel], 
                orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell), 
                cells_per_block=(cell_per_block, cell_per_block), 
                transform_sqrt=xform,
                visualise=True, 
                feature_vector=True,
                block_norm="L2-Hys",

            )
        features = features.ravel()

    return features


def load_data(car_or_not='car', ratio=1, length=-1, random=False, sanity=False, vis=False):
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
               raise Exception('Not all images have the same shape. {} was of size {}.'.format(fname, shape))
            if cur_img.dtype != data_type:
               raise Exception('Not all images have the same data type. {} was of type {}.'.format(fname, dtype))
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


def get_xy(kwargs, clf_fname=None, length=-1):
    """Load image data and extract features and labels for training

    Args:
        clf_fname: the file name where the classifier will be stored. This is used to generate
            a filename for the StandardScaler object to also be stored.
        length: the amount of data to load per car or notcar, meaning a value of 2500 will
            try to load 2500 car or notcar images depending on the value of car_or_not

    Returns:
        X matrix of features and y vector of labels.

    """

    #print('Loading features and labels...')

    car_data = load_data(length=length, car_or_not='car')
    notcar_data = load_data(length=length, car_or_not='notcar', sanity=False)

    car_features = pipeline(car_data, kwargs)
    notcar_features = pipeline(notcar_data, kwargs)

    # Create an array stack of feature vectors
    X_not_scaled = np.vstack((car_features, notcar_features)).astype(np.float32)

    scaler = StandardScaler()

    try:
        X = scaler.fit_transform(X_not_scaled)
    except ValueError as e:
        return None, None

    # Very important that the StandardScaler that's made here is reused for all future predictions
    if clf_fname is not None:
        scaler_fname = ''.join(clf_fname.split('.')[:-1]) + '_scaler.pkl'
        pickle.dump(scaler, open(scaler_fname, 'wb'))

    # Generate the labels
    car_labels = np.ones(len(car_features))
    notcar_labels = np.zeros(len(notcar_features))
    y = np.append(car_labels, notcar_labels)

    #print('Loaded, extracted features, scaled and labeled {:,} images, {:,} cars and {:,} not cars'.format(
    #    len(y), len(car_data), len(notcar_data)))
    #print('StandardScaler was pickled to {}'.format(scaler_fname))

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

    #print('Fitting data...')

    # Parameters for GridSearchCV
    grid_search = False
    parameters = [
            #{'kernel': ['linear'], 'C': [1]},
            {'kernel': ['linear'], 'C': [0.001, 0.01, 1, 10]},
            #{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.001, 0.001, 1, 10, 100]},
        ]
    if grid_search:
        clf = GridSearchCV(conf.svc(probability=True), parameters, cv=3, verbose=9)
    else:
        clf = SVC(kernel='linear', C=1, probability=True) 
        #clf = SVC(kernel='linear', C=0.1, probability=True) 

    clf.fit(X_train, y_train)

    #print(clf)
    #print('Final score on test set: {}'.format(clf.score(X_test, y_test)))
    #print()

    return clf, clf.score(X_test, y_test)


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


def pipeline(img_data, kwargs):
    features = []
    ctr = 0
    for fname, img, idx in img_data:
        assert img.max() > 1, "Pixel value range is not (0, 255), it's {}".format((img.min(), img.max()))
#        spatial_features = get_spatial(img)
#        color_features = get_colorhist(img)
        hog_features = get_hog(img, **kwargs)
        features.append(np.concatenate((
#                spatial_features, 
#                color_features, 
                hog_features,
           )).astype(np.float32))
        ctr += 1
    assert len(features) > 0, 'Got no features'
    return features


hog_candidates_csv = """
ID	tcmap	pix_per_cell	cell_per_block	orient	xform
1	GRAY	8	1	10	TRUE
2	GRAY	8	1	10	FALSE
3	GRAY	8	1	12	TRUE
4	GRAY	8	1	12	FALSE
5	GRAY	8	2	10	TRUE
6	GRAY	8	2	10	FALSE
7	GRAY	8	2	12	TRUE
8	GRAY	8	2	12	FALSE
9	GRAY	16	1	10	TRUE
10	GRAY	16	1	10	FALSE
11	GRAY	16	1	12	TRUE
12	GRAY	16	1	12	FALSE
13	GRAY	16	2	10	TRUE
14	GRAY	16	2	10	FALSE
15	GRAY	16	2	12	TRUE
16	GRAY	16	2	12	FALSE
17	RGB	8	1	10	TRUE
18	RGB	8	1	10	FALSE
19	RGB	8	1	12	TRUE
20	RGB	8	1	12	FALSE
21	RGB	8	2	10	TRUE
22	RGB	8	2	10	FALSE
23	RGB	8	2	12	TRUE
24	RGB	8	2	12	FALSE
25	RGB	16	1	10	TRUE
26	RGB	16	1	10	FALSE
27	RGB	16	1	12	TRUE
28	RGB	16	1	12	FALSE
29	RGB	16	2	10	TRUE
30	RGB	16	2	10	FALSE
31	RGB	16	2	12	TRUE
32	RGB	16	2	12	FALSE
33	HSV	8	1	10	TRUE
34	HSV	8	1	10	FALSE
35	HSV	8	1	12	TRUE
36	HSV	8	1	12	FALSE
37	HSV	8	2	10	TRUE
38	HSV	8	2	10	FALSE
39	HSV	8	2	12	TRUE
40	HSV	8	2	12	FALSE
41	HSV	16	1	10	TRUE
42	HSV	16	1	10	FALSE
43	HSV	16	1	12	TRUE
44	HSV	16	1	12	FALSE
45	HSV	16	2	10	TRUE
46	HSV	16	2	10	FALSE
47	HSV	16	2	12	TRUE
48	HSV	16	2	12	FALSE
49	YCrCb	8	1	10	TRUE
50	YCrCb	8	1	10	FALSE
51	YCrCb	8	1	12	TRUE
52	YCrCb	8	1	12	FALSE
53	YCrCb	8	2	10	TRUE
54	YCrCb	8	2	10	FALSE
55	YCrCb	8	2	12	TRUE
56	YCrCb	8	2	12	FALSE
57	YCrCb	16	1	10	TRUE
58	YCrCb	16	1	10	FALSE
59	YCrCb	16	1	12	TRUE
60	YCrCb	16	1	12	FALSE
61	YCrCb	16	2	10	TRUE
62	YCrCb	16	2	10	FALSE
63	YCrCb	16	2	12	TRUE
64	YCrCb	16	2	12	FALSE
65	YUV	8	1	10	TRUE
66	YUV	8	1	10	FALSE
67	YUV	8	1	12	TRUE
68	YUV	8	1	12	FALSE
69	YUV	8	2	10	TRUE
70	YUV	8	2	10	FALSE
71	YUV	8	2	12	TRUE
72	YUV	8	2	12	FALSE
73	YUV	16	1	10	TRUE
74	YUV	16	1	10	FALSE
75	YUV	16	1	12	TRUE
76	YUV	16	1	12	FALSE
77	YUV	16	2	10	TRUE
78	YUV	16	2	10	FALSE
79	YUV	16	2	12	TRUE
80	YUV	16	2	12	FALSE
81	LUV	8	1	10	TRUE
82	LUV	8	1	10	FALSE
83	LUV	8	1	12	TRUE
84	LUV	8	1	12	FALSE
85	LUV	8	2	10	TRUE
86	LUV	8	2	10	FALSE
87	LUV	8	2	12	TRUE
88	LUV	8	2	12	FALSE
89	LUV	16	1	10	TRUE
90	LUV	16	1	10	FALSE
91	LUV	16	1	12	TRUE
92	LUV	16	1	12	FALSE
93	LUV	16	2	10	TRUE
94	LUV	16	2	10	FALSE
95	LUV	16	2	12	TRUE
96	LUV	16	2	12	FALSE
"""

data = StringIO(hog_candidates_csv)
hgcfgs = pd.read_csv(data, sep='\t')

# Enable or disable the grid search
execute_full = True

res = [hgcfgs.columns.tolist() + ['score']]
print('{}: {}'.format(res[0][:-1], 'score'))
if execute_full:
    for i in range(len(hgcfgs)):
        kwargs = dict(hgcfgs.iloc[i][1:])
        X, y =  get_xy(kwargs, clf_fname='/var/tmp/tmp_model.pkl')
        if X is not None and y is not None:
            clf, score = get_svm_model(train=True, X=X, y=y)
            res.append(hgcfgs.iloc[i].tolist() + [score])
            print('{}: {}'.format(hgcfgs.iloc[i].tolist(), score))
        else:
            res.append(hgcfgs.iloc[i].tolist() + [None])
            print('{}: {}'.format(hgcfgs.iloc[i].tolist(), None))

    with open('./hog_param_serach.pkl', 'wb') as f:
        pickle.dump(res, f)
else:
    res = pickle.load(open('./hog_param_search.pkl', 'rb'))

df = pd.DataFrame(res[1:], columns=res[0])
df.set_index('ID')


getx = lambda color: np.where(colors == color)[0][0]*10

#@TODO: visualize the dataframe and show the best params













