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

from skimage.feature import hog

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from mpl_toolkits.mplot3d import Axes3D


CAR_FNAMES = glob.glob('./training_images/vehicles/*/*.png')
NOTCAR_FNAMES = glob.glob('./training_images/non-vehicles/*/*.png')


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
        template_list: list of image templates file names (they are read 
            from disk in this method)
        method: cv2.mathTemplate matching method. Other options include: 
            cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF, cv2.TM_CCORR, cv2.TM_SQDIFF,
            cv2.TM_SQDIFF_NORMED

    Returns:
        list of bounding boxes that were found

    """
    bbox_list = []
    for template in template_list:
        tmp = mpimg.imread(template)
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

    features = cv2.resize(img, size).ravel()

    return features


def load_data(car_or_not='car', random=False, sanity=False, vis=False):
    """Load all images from disk.

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

        image_shape = mpimg.imread(car_fnames[0]).shape
        data_type = mpimg.imread(car_fnames[0]).dtype

        for fname in car_fnames + notcar_fnames:
             shape = mpimg.imread(fname).shape
             if shape != image_shape:
                raise Exception('Not all images have the same shape. {} was of size {}.'.format(fname, shape))
             dtype = mpimg.imread(fname).dtype
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
        car_image = mpimg.imread(car_fnames[car_ind])
        notcar_image = mpimg.imread(notcar_fnames[notcar_ind])

        # Plot the examples
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(car_image)
        plt.title('Example Car Image')
        plt.subplot(122)
        plt.imshow(notcar_image)
        plt.title('Example Not-car Image')
        plt.show()

    if car_or_not == 'car':
        if random is True:
            idx = np.random.randint(0, len(car_fnames))
            return car_fnames[idx], mpimg.imread(car_fnames[idx]), idx
        else:
            # The line below returns a generator of tuples
            return ((f, mpimg.imread(f)) for f in car_fnames)
    else:
        if random is True:
            idx = np.random.randint(0, len(notcar_fnames))
            return notcar_fnames[idx], mpimg.imload(notcar_fnames[idx]), idx
        else:
            # The line below returns a generator of tuples
            return ((f, mpimg.imread(f)) for f in notcar_fnames)


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


def extract_features(img_gnrtr, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), vis=False):
    """Extract features using bin_spatial() and color_hist() to generate a feature vector.

    Args:
        img_gnrtr: a generator of RGB images
        cspace: color space to pass to `bin_spatial()`
        spacial_size: input to be used for the `bin_spacial()` function
        hist_bins: number of bins for the `color_hist()` function
        hist_range: hist range to pass to the `color_hist()` function
        vis: if True, plot the results in a window

    Returns:
        single list of normalized features

    """
    features = []

    for fname, img in img_gnrtr:
        spatial_features = bin_spatial(img, color_space=cspace, size=spatial_size)
        color_features = color_hist(img, nbins=hist_bins, bins_range=hist_range)
        features.append(np.concatenate((spatial_features, color_features)))

        if vis is True:
            fig = plt.figure(figsize=(12,4))

            plt.subplot(131)
            plt.imshow(img)
            plt.title('Original Image')

            plt.subplot(132)
            plt.plot(spatial_features)
            plt.title('Spatial')

            plt.subplot(133)
            plt.plot(color_features)
            plt.title('Color')

            fig.tight_layout()
            plt.show()

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
    
    # Helper lambda to easily convert to grayscale and other conversions
    rgb2gray = lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bgr2rgb = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ###############################################################
    #
    # Create a Support Vector Machine
    #
    ###############################################################
    #clf = svm()


    ###############################################################
    #
    # Test adding boundary boxes
    #
    ###############################################################
    #image = mpimg.imread('./test_images/bbox-example-image.jpg')
    #bboxes = [((837, 507), (1120, 669))]
    #result = draw_boxes(image, bboxes)
    #plt.imshow(result)
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
    #image = cv2.imread('./test_images/bbox-example-image.jpg')
    #image = draw_boxes(image, bboxs)
    #plt.imshow(image)
    #plt.show()


    ###############################################################
    #
    # Show color histogram for RGB space
    #
    ###############################################################
    #img = mpimg.imread('test_images/cutout1.jpg')
    #rhist, ghist, bhist, hist_features, bin_centers = color_hist(img)
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
    #    img = cv2.imread(f)
    #    
    #    # Select a small fraction of pixels to plot by subsampling it
    #    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    #    img_small = cv2.resize(img, (np.int(img.shape[1] / scale),
    #        np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
    #    
    #    # Convert subsampled image to desired color space(s)
    #    img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    #    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    #    img_small_HLS = cv2.cvtColor(img_small, cv2.COLOR_BGR2HLS)
    #    img_small_YCR = cv2.cvtColor(img_small, cv2.COLOR_BGR2YCrCb)
    #    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
    #    
    #    # Plot and show
    #    #plot3d(img_small_RGB, img_small_rgb)
    #    #plt.show()
    #    #
    #    #plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    #    #plt.show()

    #    #plot3d(img_small_HLS, img_small_rgb, axis_labels=list("HLS"))
    #    #plt.show()

    #    plot3d(img_small_YCR, img_small_rgb, axis_labels=list("HLS"))
    #    plt.show()


    ###############################################################
    #
    # Resizing images
    #
    ###############################################################

    #image = mpimg.imread('test_images/cutout1.jpg')
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
    #fname, image, idx = load_data(car_or_not='car', random=True)

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
    #plt.imshow(image, cmap='gray')
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


    ###############################################################
    #
    # Create the image loading generators (fname, rgb_img) tuples
    #
    ###############################################################

    car_gnrtr = load_data(car_or_not='car')
    notcar_gnrtr = load_data(car_or_not='notcar')


    ###############################################################
    #
    # Extract features
    #
    ###############################################################

    car_features = extract_features(car_gnrtr, cspace='RGB', spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256))
    notcar_features = extract_features(notcar_gnrtr, cspace='RGB', spatial_size=(32, 32),
                                    hist_bins=32, hist_range=(0, 256))

    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        
        # Plot an example of raw and scaled features
        fname, random_car, idx = load_data(car_or_not='car', random=True)
        fig = plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(random_car)
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[idx])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[idx])
        plt.title('Normalized Features')
        fig.tight_layout()
        plt.show()
    else: 
        print('Your function only returns empty feature vectors...')


if __name__ == '__main__':
    main()


