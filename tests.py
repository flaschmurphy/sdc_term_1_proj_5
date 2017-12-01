import pickle
import glob
from obj_detection import *
import matplotlib.pyplot as plt


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

    with open('model1.pkl', 'rb') as f:
        clf = pickle.load(f)

    for fname in car_fnames:
        predict(fname, clf, car=True)

    for fname in notcar_fnames:
        predict(fname, clf, car=False)

def main():
    test_predictions()


if __name__ == '__main__':
    main()


