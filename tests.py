import pickle
import glob
import cv2
import matplotlib.pyplot as plt

from obj_detection import *

MODEL_FILE = './model_full.pkl'

def predict(fname, clf, car=True, vis=True):
    """Run the classifier on an image and get a prediction"""

    with open(fname, 'rb') as f:
        img = imread(f)

    features = extract_features([[fname, img, 0]], cspace='YCrCb',
            vis=vis)
    prediction = int(clf.predict(features)[0])
    confidence = clf.decision_function(features)[0]

    truth = 'car'
    if prediction == 0:
        truth = 'notcar'

    result = 'CORRECT'
    if prediction == 0 and car:
        result = 'INCORRECT'
    if prediction == 1 and not car:
        result = 'INCORRECT'
    print('{:35}, model predicts {} ({:6}) --> {:9} ({})'.format(
        fname, prediction, truth, result, confidence))
 

def test_predictions():
    """Run a sequence of images through `predict()`"""

    car_fnames = glob.glob('./test_images/car_*')
    notcar_fnames = glob.glob('./test_images/notcar_*')

    with open(MODEL_FILE, 'rb') as f:
        clf = pickle.load(f)

    for fname in car_fnames:
        predict(fname, clf, car=True)

    for fname in notcar_fnames:
        predict(fname, clf, car=False)


def test_windowing():
    img_size = (960, 1280, 3)
    window_size = (64, 64)
    overlap = 0.5
    start_pos = (0, 0)

    get_window_points(img_size, window_size, overlap, start_pos, vis=True)


def main():
    #test_predictions()
    test_windowing()


if __name__ == '__main__':
    main()


