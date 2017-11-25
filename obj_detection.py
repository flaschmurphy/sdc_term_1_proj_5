"""
 Udacity Self Driving Car Nanodegree

 Term 1, Project 5 -- Object Detection

 Author: Ciaran Murphy
 Date: 27th Nov 2017

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def svm(X=None, y=None, kernel='linear', C=1.0, gamma='auto'):
    """Create a SVM.

    Args:
        X: training features
        y: training labels
        kernel: the tupe of SVM kernel to use
        C: penalty parameter C of the error term
        gamma: defines how far the influence of a single training example reaches

    Returns:
        if X and y are not None, a trained SVM, otherwise an untrained SVM

    """
    clf = SVC(C=C, kernel=kernel, gamma=gamma)

    if X is not None and y is not None:
        clf.fit(X, y)

    return clf


def main():
    """The main entry point"""
    s = svm()


if __name__ == '__main__':
    main()


