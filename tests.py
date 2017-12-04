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
from argparse import ArgumentParser

from obj_detection import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--classifier', dest='clf', required=True,
            help='The location of the pickled classifier')

    args = parser.parse_args()
    return args


def test_predictions():
    """Test a sequence of images on `predict()`."""
    print()
    print('Testing `predict()`')

    car_fnames = glob.glob('./test_images/car_*')
    notcar_fnames = glob.glob('./test_images/notcar_*')

    with open(args.clf, 'rb') as f:
        clf = pickle.load(f)

    for fname in car_fnames:
        img = imread(fname)
        predict(img, clf, scaler_fname, fname=fname, car=True, vis=True)

    for fname in notcar_fnames:
        img = imread(fname)
        predict(img, clf, scaler_fname, fname=fname, car=False, vis=True)

    plt.pause(2)
    plt.close()

    print()


def test_windowing():
    """Test boxes returned from `get_window_points()`. """
    print()
    #
    # Far boxes
    #
    print('Testing get_window_points() for far boxes...')
    img_size = (720, 1216, 3)
    window_size = (64, 64)
    overlap = 0.5
    start_pos = (200, 400)
    end_pos = (1180, 464)
    bboxes_far = get_window_points(img_size, window_size, overlap, start=start_pos, end=end_pos)
    assert len(bboxes_far) == 29, 'Incorrect number of windows, got {}, should be 29.'.format(
            len(bboxes_far))

    #
    # Middle boxes
    #
    print('Testing get_window_points() for middle boxes...')
    img_size = (720, 1280, 3)
    window_size = (306, 128)
    overlap = 0.5
    start_pos = (35, 428)
    end_pos = (1280, 592)
    bboxes_middle = get_window_points(img_size, window_size, overlap, start=start_pos, end=end_pos)
    #assert len(bboxes_middle) == 14, 'Incorrect number of windows, got {}, should be 14.'.format(
    #        len(bboxes_middle))

    #
    # Near boxes
    #
    print('Testing get_window_points() for near boxes...')
    img_size = (720, 1280, 3)
    window_size = (306, 256)
    overlap = 0.5
    start_pos = (35, 364)
    end_pos = (1280, 650)
    bboxes_near = get_window_points(img_size, window_size, overlap, start=start_pos, end=end_pos)
    #assert len(bboxes_near) == 46, 'Incorrect number of windows, got {}, should be 46.'.format(
    #        len(bboxes_near))
    print(len(bboxes_near))

    # Visualize the results
    vis = True
    if vis:
        #fname = './test_images/bbox-example-image.jpg'
        fname = './test_frames/frame_031.png'
        get_clr = lambda: np.random.randint(255, size=3).tolist()

        plt.close()
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 10))

        canvas1 = imread(fname)
        for box in bboxes_far:
            cv2.rectangle(canvas1, box[0], box[1], get_clr(), 3)
        imshow(canvas1, axis=ax1)

        canvas2 = imread(fname)
        for box in bboxes_middle:
            cv2.rectangle(canvas2, box[0], box[1], get_clr(), 3)
        imshow(canvas2, axis=ax2)

        canvas3 = imread(fname)
        for box in bboxes_near:
            cv2.rectangle(canvas3, box[0], box[1], get_clr(), 3)
        imshow(canvas3, axis=ax3)

        plt.tight_layout()
        plt.show()

        plt.pause(2)
        plt.close()

    print()


def test_sliding_window_predictions(fname=None):
    """Test that a sliding window slides as expected and detects cars"""
    #
    # Run a scan over an image and retain any boxes where a car is detected
    #
    if fname is None:
        #img = imread('./test_images/bbox-example-image.jpg')
        img = imread('./test_frames/frame_040.png')
    else:
        img = imread(fname)
        print('Loaded {} with shape {}'.format(fname, img.shape))


    #
    # Run far bounding boxes
    #
    window_size = (64, 64)
    overlap = 0.25
    start_pos = (200, 400)
    end_pos = (1180, 464)
    bboxes_far = get_window_points(img.shape, window_size, overlap, start=start_pos, end=end_pos)

    for box in bboxes_far:
        top_left, bottom_right = box
        sub_img = img[ top_left[1]: bottom_right[1], top_left[0]: bottom_right[0], :]

    #
    # Run middle bounding boxes
    #
    window_size = (306, 100)
    overlap = 0.25
    start_pos = (35, 390)
    end_pos = (1280, 592)
    bboxes_mid = get_window_points(img.shape, window_size, overlap, start=start_pos, end=end_pos)

    for box in bboxes_mid:
        top_left, bottom_right = box
        sub_img = img[ top_left[1]: bottom_right[1], top_left[0]: bottom_right[0], :]

    #
    # Run near bounding boxes
    #
    window_size = (275, 225)
    overlap = 0.5
    start_pos = (35, 364)
    end_pos = (1280, 650)
    bboxes_near = get_window_points(img.shape, window_size, overlap, start=start_pos, end=end_pos)

    for box in bboxes_near:
        top_left, bottom_right = box
        sub_img = img[ top_left[1]: bottom_right[1], top_left[0]: bottom_right[0], :]


    # If True, show realtime visualizations
    vis = True    
    if vis:
        display_img = img.copy()    # Use this one for showing results
        plt.ion()    # Enable interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10)) 

    # Setup a CPU timer to find out how much time this all takes
    start = time.clock()

    # Load the classifier
    clf = get_svm_model(model_file=args.clf)

    get_clr = lambda: np.random.randint(255, size=3).tolist()

    cnt = 0
    car_boxes = []
    for box in bboxes_far + bboxes_mid + bboxes_near:
        top_left, bottom_right = box
        sub_img = img[ top_left[1]: bottom_right[1], top_left[0]: bottom_right[0], :]

        prediction = predict(sub_img, clf, scaler_fname, 
                fname='./test_images/bbox-example-image.jpg', vis=False, verbose=True)

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

            imshow(display_img_, axis=ax1)
            imshow(sub_img, axis=ax2)

            # Uncomment below to save frames while running
            #new_fname = './training_images/my_images/notcar/{}_{:03}_2.png'.format(
            #        os.path.basename(fname).split('.')[0], cnt)
            #imsave(new_fname, sub_img)

            plt.tight_layout()
            plt.show()
            plt.pause(0.0001)

            cnt += 1

    end = time.clock()
    duration = end - start

    plt.sca(ax1)
    plt.cla()
    ax1.imshow(display_img[:,:,::-1]/255)
    plt.show()
    plt.pause(1)
    plt.close()

    print('Prcessing one image took {:.3}s in CPU time.'.format(duration))

        
def plot_example_features():
    """"Plot an example of raw and scaled features"""
    enable=False

    num_plots = 5
    if enable:
        for i in range(num_plots):
            fname, image, idx = load_data(car_or_not='car', random=True)[0]

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


def main():
    global args, scaler_fname

    args = parse_args()
    scaler_fname = ''.join(args.clf.split('.')[:-1]) + '_scaler.pkl'

    plot_example_features()
    test_predictions()
    test_windowing()
    test_sliding_window_predictions()
    for fname in glob.glob('./test_frames/*'):
        test_sliding_window_predictions(fname)

    pass

if __name__ == '__main__':
    main()

