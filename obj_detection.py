"""
 Udacity Self Driving Car Nanodegree

 Term 1, Project 5 -- Object Detection

 Author: Ciaran Murphy
 Date: 27th Nov 2017

"""

import glob
import cv2
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from skimage.feature import hog

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from mpl_toolkits.mplot3d import Axes3D


CAR_FNAMES = glob.glob('./training_images/vehicles/*/*.png')
NOTCAR_FNAMES = glob.glob('./training_images/non-vehicles/*/*.png')


def imread(fname):
    """Helper function to always load images in a consistent way"""
    img = mpimg.imread(fname).astype(np.float32)
    if '.png' in fname:
        return img * (255.0/img.max())
    return img


def imshow(*args, **kwargs):
    """Helper function to always show images in a consistent way"""
    args_ = [i for i in args]
    imgc = args_[0].copy()
    if imgc.max() > 10:
        imgc = (imgc - imgc.min()) / (imgc.max() - imgc.min())
    args_[0] = imgc
    plt.imshow(*args_, **kwargs)


def svm(X=None, y=None, kernel='linear', C=1.0, gamma='auto'):
    """Create a Support Vector Machine. Advantages of SVMs - the work well in
    complex domains where there is a clear margin of separation. Don't work
    well where there is a lot of data because the training is cubic in the
    size of the dataset. Also don't work in the presence of a lot of noise (try
    Naive Bayes if there's lots of noise).

    Args:
        X: training features
        y: training labels
        kernel: the tupe of SVM kernel to use
        C: controls the tradeoff between a smooth deision boundary and
            classifying training points correctly. A large C means you'll get more
            training points correct and a more complex decision boundary
        gamma: defines how far the influence of a single training example
            reaches. Low value will mean that every point has a far reach, high
            values means points only have a close reach.

    Returns:
        if X and y are not None, a trained SVM, otherwise an untrained SVM

    """
    clf = SVC(C=C, kernel=kernel, gamma=gamma)

    if X is not None and y is not None:
        clf.fit(X, y)

    return clf


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """Draw a rectangular box over an image

    Args:
        img: the image to draw over
        bboxes: list of box points, [(top left, bottom right), ...]
        color: RGB color
        thick: thickness

    Returns:
        a copy of the original image with the boxes added

    """
    draw_img = np.copy(img)

    for b in bboxes:
        cv2.rectangle(draw_img, b[0], b[1], color, thick)

    return draw_img 


def find_matches(img, template_fnames, method=cv2.TM_CCOEFF_NORMED):
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


def color_hist(img, nbins=32, bins_range=(0, 256), vis=False):
    """Generate a histogram of color channels (RGB).
    Args:
        img: the imput image
        nbins: the number of bins in the histogram
        bins_range: the range for the bins
        vis: if True, output data for creating a visualization

    Returns:
        red histogram
        green histogram 
        blue histogram 
        hist_features (contatenated r,g,b histograms)
        bin_centers

    """
    ch1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    ch2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    ch3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((ch1_hist[0], ch2_hist[0], ch3_hist[0]))

    if vis is False:
        return hist_features

    bin_edges = ch1_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges)-1])/2

    return ch1_hist, ch2_hist, ch3_hist, hist_features, bin_centers


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


def bin_spatial(img, color_space='RGB', size=(32, 32)):
    """Convert an image to the provided color space, rescale it and unroll it.

    Args:
        img: the input image (assumed to be in RGB color space)
        color_space: target color space to convert to
        size: target size to scale to

    Returns:
        one dimensional feature vector of the converted image

    """
    assert color_space in ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb'], "Invalid target color space."

    if color_space == 'BGR':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2RGB)
    elif color_space == 'HSV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif color_space == 'HLS':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'YUV':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCrCb':
        feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: 
        feature_image = np.copy(img)

    features = cv2.resize(feature_image, size).ravel()

    return features


def load_data(car_or_not='car', random=False, sanity=False, vis=False):
    """Load all images from disk. Loads using matplotlib --> range will be 0 to 1.

    Args:
        car_or_not: if 'car', return a generator of car images, if 'not', return a generator 
            for images that are not of cars
        random: if True, instead of returning a generator for images, return a single fname
            and image selected at random
        sanity: if True print out some info about the data and verify it's consistency
        vis: if True, display a random sample image

    Returns:
       a generator of 2-tuples containing (filename, image). Note: the color space is RGB.

    """
    assert car_or_not in ['car', 'notcar'], "Invalid option for 'car_or_not'. Chose from ['car', 'notcar']"

    car_fnames = CAR_FNAMES
    notcar_fnames = NOTCAR_FNAMES

    # Verify that all images are of the same shape and dtype
    if sanity:
        print('Checking that all images have the same size and dtype...')

        image_shape = imread(car_fnames[0]).shape
        data_type = imread(car_fnames[0]).dtype

        for fname in car_fnames + notcar_fnames:
             shape = imread(fname).shape
             if shape != image_shape:
                raise Exception('Not all images have the same shape. {} was of size {}.'.format(fname, shape))
             dtype = imread(fname).dtype
             if dtype != data_type:
                raise Exception('Not all images have the same data type. {} was of type {}.'.format(fname, dtype))
        print('  All images are consistent!')
 
        data = {
               'n_notcars': len(notcar_fnames), 
               'n_cars': len(car_fnames),
               'image_shape': image_shape,
               'data_type': data_type,
               }

        print('  Total number of images: {} cars and {} non-cars'.format(data["n_cars"], data["n_notcars"]))
        print('  Image size: {}, data type: {}'.format(data["image_shape"], data["data_type"]))

    if vis:
        # Choose random car / not-car indices and plot example images   
        car_ind = np.random.randint(0, len(car_fnames))
        notcar_ind = np.random.randint(0, len(notcar_fnames))
            
        # Read in car / not-car images
        car_image = imread(car_fnames[car_ind])
        car_image *= (255.0/car_image.max())    # normalize to (0, 255)

        notcar_image = imread(notcar_fnames[notcar_ind])

        # Plot the examples
        fig = plt.figure()
        plt.subplot(121)
        imshow(car_image)()
        plt.title('Example Car Image')
        plt.subplot(122)
        imshow(notcar_image)()
        plt.title('Example Not-car Image')
        plt.show()

    if car_or_not == 'car':
        if random is True:
            idx = np.random.randint(0, len(car_fnames))
            img = imread(car_fnames[idx])
            yield (car_fnames[idx], img, idx)
        else:
            # returns a generator of tuples
            for f in car_fnames:
                yield f, imread(f)
    else:
        if random is True:
            idx = np.random.randint(0, len(notcar_fnames))
            img = imread(notcar_fnames[idx])
            yield notcar_fnames[idx], img, idx
        else:
            # returns a generator of tuples
            for f in notcar_fnames:
                yield f, imread(f)


def get_hog_features(img, orient=9, pix_per_cell=9, cell_per_block=2, vis=False, feature_vec=True):
    """Get a Histogram of Oriented Gradients for an image.

    "Note: you could also include a keyword to set the tranform_sqrt flag but
    for this exercise you can just leave this at the default value of
    transform_sqrt=False"

    Args:
        img: a single channel (or gray) image
        orient: the number of orientation bins in the output histogram. Typical values are between 6 and 12
        pix_per_cell: a 2-tuple giving the number of pixels per cell e.g. `(9, 9)`
        cells_per_block: a 2-tuple giving the number of cells per block
        vis: if True, return a visualization of the HOG
        feature_vec: if True, return the unroll the features vector before returning it

    Returns:
        Features vector if visualize is False, or (features vector, hog_image) otherwise.
        If feature_vec is True, the returned features will be already unrolled.

    """
    ret = hog(
            img, 
            orientations=orient,
            pixels_per_cell=(pix_per_cell, pix_per_cell), 
            cells_per_block=(cell_per_block, cell_per_block), 
            transform_sqrt=False,
            visualise=vis, 
            feature_vector=feature_vec,
            block_norm="L2-Hys"
        )

    if vis is True:
        features, hog_image = ret
        if feature_vec:
            features = features.ravel()
        return features, hog_image
    else:
        if feature_vec:
            features = ret.ravel()
        return features


def extract_features(img_gnrtr, cspace='RGB', spatial_size=(32, 32), hist_bins=32, 
        hist_range=(0, 256), vis=False, length=-1):
    """Extract features using bin_spatial() and color_hist() to generate a feature vector.

    Args:
        img_gnrtr: a generator of RGB images
        cspace: color space to pass to `bin_spatial()`
        spacial_size: input to be used for the `bin_spacial()` function
        hist_bins: number of bins for the `color_hist()` function
        hist_range: hist range to pass to the `color_hist()` function
        vis: if True, plot the results in a window
        length: if >-1, return a subset of the data that long, otherwise return all

    Returns:
        single list of normalized features

    """
    features = []

    if length == -1:
        length = float('inf')

    ctr = 0
    for fname, img in img_gnrtr:
        spatial_features = bin_spatial(img, color_space=cspace, size=spatial_size)
        assert len(spatial_features) > 0, 'Got no data back from bin_spatial,'

        hist_features = color_hist(img, nbins=hist_bins, bins_range=hist_range)
        assert len(hist_features) > 0, 'Got no data back from color_hist,'

        features.append(np.concatenate((spatial_features, hist_features)).astype(np.float32))

        if vis is True:
            fig = plt.figure(figsize=(12,4))

            plt.subplot(131)
            imshow(img)()
            plt.title('Original Image')

            plt.subplot(132)
            plt.plot(spatial_features)
            plt.title('Spatial')

            plt.subplot(133)
            plt.plot(color_features)
            plt.title('Color')

            fig.tight_layout()
            plt.show()

        ctr += 1
        if ctr == length:
            break

    assert len(features) > 0, 'Got no features.'
    return features


def main():
    """The main entry point

    Reminders: 

    * In the HOG function, you could also include a keyword to set the tranform_sqrt flag but
      for this exercise you can just leave this at the default value of
      transform_sqrt=False

    * Don't forget to normalize any concatenated features in the pipeline (they should 
      all be in the same range, e.g. between 0 and 1). Use sklearn.StandardScalar()

    * Either use cv2.imread (meaning BGR) or mpimg.imread (meaning RGB) but don't mix them

    * Feature extraction: 
       - Raw pixel intensity:              color and shape
       - Histogram of pixel intensity:     color only
       - Gradients of pixel intensity:     shape only

    """
    ###############################################################
    #
    # Extract features and labels
    #
    ###############################################################
    car_gnrtr = load_data(car_or_not='car')
    notcar_gnrtr = load_data(car_or_not='notcar')

    print('Loading all features and labels...')
    car_features = extract_features(car_gnrtr, cspace='RGB', spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256), length=-1)
    notcar_features = extract_features(notcar_gnrtr, cspace='RGB', spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256), length=-1)

    # Create an array stack of feature vectors
    X_not_scaled = np.vstack((car_features, notcar_features)).astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_not_scaled)

    # Generate the labels
    car_labels = np.ones(len(car_features))
    notcar_labels = np.zeros(len(notcar_features))
    y = np.append(car_labels, notcar_labels)

    print('Loaded, extracted features, scaled and labeled {:,} images, {:,} cars and {:,} not cars'.format(
        len(y), len(car_features), len(notcar_features)))


    ###############################################################
    #
    # Plot an example of raw and scaled features
    #
    ###############################################################
    num_plots = 5
    enable=False
    if enable:
        for i in range(num_plots):
            for t in load_data(car_or_not='car', random=True):
                fname, image, idx = t
            fig = plt.figure(figsize=(12,4))

            plt.subplot(131)
            imshow(image)
            plt.title('Original Image')

            plt.subplot(132)
            plt.plot(X[idx])
            plt.title('Raw Features')

            plt.subplot(133)
            plt.plot(X_scaled[idx])
            plt.title('Scaled features')

            fig.tight_layout()
            plt.show()

    ###############################################################
    #
    # Create a Support Vector Machine and GridSearchCV 
    # for optimal params.
    #
    ###############################################################

    # Set the amount of data to use for training the classifier.
    subset_size = 100

    print('Starting GridSearchCV on subset of length {}...'.format(subset_size))

    #parameters = {'kernel': ('linear', 'rbf'), 'C': range(1, 10)}
    parameters = [
            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        ]

    #scores = ['precision', 'recall', 'accuracy']
    scores = ['accuracy']

    idx = np.random.permutation(X.shape[0])[:subset_size]
    X_subset = X[idx]
    y_subset = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=42, shuffle=True)

    for score in scores:
        clf = GridSearchCV(SVC(C=1), parameters, cv=5, scoring=score, verbose=1)
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        print(clf.best_score_)


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
    #
    ###############################################################


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
    # Show color histogram for RGB space
    #
    ###############################################################
    #img = imread('test_images/cutout1.jpg')
    #rhist, ghist, bhist, hist_features, bin_centers = color_hist(img, vis=True)
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
    # Explore the images
    #
    ###############################################################
    #load_data(sanity=True)


    ###############################################################
    #
    # Test the HOG function
    #
    ###############################################################

    # Generate a random index to look at a car image
    #for t in load_data(car_or_not='car', random=True):
    #    fname, image, idx = t

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

