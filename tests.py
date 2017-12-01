"""
 Udacity Self Driving Car Nanodegree

 Term 1, Project 5 -- Tests for Object Detection

 Author: Ciaran Murphy
 Date: 27th Nov 2017

"""
import pickle
import glob
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec

from obj_detection import *

MODEL_FILE = './model_full.pkl'


def test_predictions():
    """Test a sequence of images on `predict()`."""

    print()
    print('Testing `predict()`')

    car_fnames = glob.glob('./test_images/car_*')
    notcar_fnames = glob.glob('./test_images/notcar_*')

    with open(MODEL_FILE, 'rb') as f:
        clf = pickle.load(f)

    for fname in car_fnames:
        with open(fname, 'rb') as f:
            img = imread(f)
        predict(img, clf, fname=fname, car=True, vis=True)

    for fname in notcar_fnames:
        with open(fname, 'rb') as f:
            img = imread(f)
        predict(img, clf, fname=fname, car=False, vis=True)


def test_windowing():
    """Test boxes returned from `get_window_points()`. """

    print()
    print('Testing `get_window_points()`')

    img_size = (960, 1280, 3)
    window_size = (64, 64)
    overlap = 0.5
    start_pos = (0, 0)

    bboxes = get_window_points(img_size, window_size, overlap, start_pos, vis=True)
    assert len(bboxes) == 1131, 'Incorrect number of windows, got {}, should be 1131.'.format(
            len(bboxes))


def test_sliding_window():
    """Test classifier on sliding windows"""
    img = imread('./test_images/bbox-example-image.jpg')
    bboxes = get_window_points(img.shape, vis=False)
    get_clr = lambda: np.random.randint(255, size=(3))

    for box in bboxes:
        cv2.rectangle(img, *box, color=np.random.randint(255))

    #imshow(img)
    #plt.show()

    # Refresh the original image (remove previously drawn boxes)
    img = imread('./test_images/bbox-example-image.jpg')
    display_img = img.copy()    # Use this one for showing results

    plt.ion() # Enable interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10)) 

    # Load the classifier
    #clf = get_svm_model(model_file='./model1.pkl')
    clf = get_svm_model(model_file='./model_full.pkl')

    for box in bboxes:
        top_left, bottom_right = box
        sub_img = img[ top_left[1]: bottom_right[1], top_left[0]: bottom_right[0], :]

        prediction, confidence = predict(sub_img, clf, fname='./test_images/bbox-example-image.jpg', vis=False)

        # The predictions are so bad that inverting them makes it better!
        prediction = int(not prediction)

        display_img_ = display_img.copy()
        cv2.rectangle(display_img_, *box, 255, 6)

        # If a car was predicted, keep the box painted
        if prediction == 1:
            display_img =  display_img_
        
        plt.sca(ax1)
        plt.cla()
        plt.sca(ax2)
        plt.cla()

        # For some reason I need to normalize the image now
        display_img_ = (display_img_ - display_img_.min()) / (display_img_.max() - display_img_.min())
        sub_img_ = (sub_img - sub_img.min()) / (sub_img.max() - sub_img.min())

        imshow(display_img_, axis = ax1)
        imshow(sub_img_, axis = ax2)
        plt.tight_layout()
        plt.show()
        plt.pause(0.001)

        
def main():
    #test_predictions()
    #test_windowing()
    test_sliding_window()


if __name__ == '__main__':
    main()


