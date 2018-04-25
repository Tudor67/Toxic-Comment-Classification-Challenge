import numpy as np
import pandas as pd

from constants import *
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def svm_classifier(train_x, train_y, test_x, method='LinearSVC'):
    #scaler = preprocessing.StandardScaler(with_mean=False).fit(train_x)
    #train_x = scaler.transform(train_x)
    #test_x = scaler.transform(test_x)

    pred = np.zeros((test_x.shape[0], len(target_classes)))

    for i in range(len(target_classes)):
        print('class {}'.format(i))
        classifier = None
        if method == 'LinearSVC':
            classifier = svm.LinearSVC()
        elif method == 'SVC':
            classifier = svm.SVC(C=1.3)

        classifier.fit(train_x, train_y[:,i])
        pred[:,i] = classifier.predict(test_x)

    return pred


def nb_classifier(train_x, train_y, test_x, method='BR'):
    classifier = None
    if method == 'BR':
        classifier = BinaryRelevance(GaussianNB())
    elif method == 'CC':
        classifier = ClassifierChain(GaussianNB())
    elif method == 'LP':
        classifier = LabelPowerset(GaussianNB())

    classifier.fit(train_x, train_y)

    pred = classifier.predict(test_x)

    return pred


def multinomial_NB_classifier(train_x, train_y, test_x):
    preds = np.zeros((test_x.shape[0], len(target_classes)))

    for i in range(len(target_classes)):
        print('step: {}'.format(i))
        classifier = MultinomialNB()
        print('---fit---')
        classifier.fit(train_x, train_y[:,i])
        print('---predict---')
        preds[:,i] = classifier.predict_proba(test_x)[:,1]
    
    return preds


def logistic_regression(train_x, train_y, test_x):
    preds = np.zeros((len(test_text), len(target_classes)))

    for i in range(len(target_classes)):
        print('step: {}'.format(i))
        lr_model = LogisticRegression(C=4, dual=True, class_weight='balanced')
        print('---fit---')
        lr_model.fit(train_x, train_y[:,i])
        print('---predict---')
        preds[:,i] = lr_model.predict_proba(test_x)[:,1]

    return preds


def logistic_regression_with_word_embeddings(train_x, train_y, test_x):
    print('---logistic_regression (word_embbedings)---')
    preds = np.zeros((len(test_x), len(target_classes)))

    for i in range(len(target_classes)):
        print('step: {}'.format(i))
        lr_model = LogisticRegression(C=4, dual=True, class_weight='balanced')
        print('---fit---')
        lr_model.fit(train_x, train_y[:,i])
        print('---predict---')
        preds[:,i] = lr_model.predict_proba(test_x)[:,1]

    return preds


def classification(classifier='NB'):
    preds = []
    if classifier == 'SVM':
        print('---SVM---')
        preds = svm_classifier(train_x, train_y, test_x, method='LinearSVC')
    elif classifier == 'NB':
        print('---multinomial_nb_classifier---')
        preds = multinomial_NB_classifier(train_x, train_y, test_x)
    else:
        print('---logistic_regression---')


    return preds