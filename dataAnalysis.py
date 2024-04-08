import panda as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from math import sqrt
from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

OUTPUT_TEMPLATE_CLASSIFIER =  (
    'bayesClassifier:   {bayes:.3g}\n'
    'kNNClassifier:     {knn:.3g}\n'
    'SVMClassifier:     {svm:.3g}\n'
)

OUTPUT_TEMPLATE_REGRESS = (
    'linearRegression: {linReg:.2g}\n'
    'polynomialRegression: {polyReg:.3g}\n'
)


# Used for the ML model for prediciting participants gender, height, weight
def ML_classifier(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    bayesModel = GaussianNB()
    knnModel = KNeighborsClassifier(n_neighbors=3)
    svcModel = SVC(kernel = 'linear')

    models = [bayesMode, knnModel, svcModel]

    for i, m in enumerate(models):
        m.fit(X_train, y_train)

    print(OUTPUT_TEMPLATE_CLASSIFIER.format(
        bayes = bayesModel.score(X_train, y_train),
        knn = knnModel.score(X_train, y_train),
        svcModel = svcModel.score(X_train, y_train)
    ))
