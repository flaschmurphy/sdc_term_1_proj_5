import pickle
import glob
from obj_detection import *
import matplotlib.pyplot as plt


def predict(fname, clf, car=True, vis=True):
    with open(fname, 'rb') as f:
        img = imread(f)

    features = extract_features([[fname, img, 0]], cspace='YCrCb',
            vis=vis)
    prediction = int(clf.predict(features)[0])

    result = 'CORRECT'
    if prediction == 0 and car:
        result = 'INCORRECT'
    if prediction == 1 and not car:
        result = 'INCORRECT'
    print('For {:30}, model predicts {} --> {}'.format(
        fname, prediction, result))
 

car_fnames = glob.glob('./test_images/car_*')
notcar_fnames = glob.glob('./test_images/notcar_*')

with open('model1.pkl', 'rb') as f:
    clf = pickle.load(f)

for fname in car_fnames:
    predict(fname, clf, car=True)

for fname in notcar_fnames:
    predict(fname, clf, car=False)


