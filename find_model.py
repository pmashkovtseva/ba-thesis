#!/usr/bin/env python3
# Author: Vera Bernhard // minor changes by Polina Mashkovtseva
# Date: 21.06.2021 // 31.05.2023

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB

import numpy as np

from stress_detector import StressDetector


# This file describes the steps of finding the best model


def get_best_classifiers(wav_path: str, features: list, classifier_names: list):
    """Check which 4 algorithms perform best, with and without post-processing"""
    sd = StressDetector(wav_path, features)
    sd.get_features('./data/complete_features.tsv')

    classifiers = [
        KNeighborsClassifier(
            n_jobs=-1
        ),
        LogisticRegression(),
        SVC(probability=True,
            random_state=42),
        DecisionTreeClassifier(
            random_state=42),
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1),
        MLPClassifier(
            random_state=42),
        AdaBoostClassifier(
            random_state=42),
        GaussianNB()]

    # with post-processing
    results_post = (sd.test_classifiers(classifiers, classifier_names)).sort_values('f1')

    print(f"With Post-Processing:\n {results_post}")

    # ==> Best performing models: Nearest Neighbour, SVM, Random Forest, Neural Net


def train_best_model(wav_path, features):
    """Train best system on all data and save it"""

    mlp_abs_cont = MLPClassifier(
        random_state=42,
        max_iter=300,
        # hyperparameters found by gridsearch
        activation='relu',
        alpha=0.0001,
        hidden_layer_sizes=(100, 50),
        solver='adam'
    )

    nn_abs_cont = KNeighborsClassifier(
        n_jobs=-1,
        # hyperparameters found by gridsearch
        algorithm='auto',
        metric='manhattan',
        n_neighbors=3,
        weights='distance'
    )

    svm_abs_cont = SVC(
        random_state=42,
        probability=True,
        # hyperparameters found by gridsearch
        C=10.0,
        class_weight=None,
        gamma='scale',
        kernel='rbf'
    )

    rf_abs_cont = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        # hyperparameters found by gridsearch
        class_weight='balanced',
        criterion='entropy',
        max_depth=50,
        min_samples_split=5,
        n_estimators=200
    )

    vot_abs_cont = VotingClassifier(
        estimators=[('mlp', mlp_abs_cont), ('nn', nn_abs_cont),
                    ('svm', svm_abs_cont), ('rf', rf_abs_cont)],
        voting='soft')

    sd = StressDetector(wav_path, features)
    sd.get_features('./data/complete_features.tsv')
    sd.train_all(vot_abs_cont, 'vot', save=True)
    evaluation = sd.train(vot_abs_cont, features, matrix=True)
    print('F1 Score: {}'.format(np.mean(evaluation['f1'])))
    print('Accuracy: {}'.format(np.mean(evaluation['accuracy'])))
