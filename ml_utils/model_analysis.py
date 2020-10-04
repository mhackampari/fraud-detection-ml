import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate


class ModelAnalyser():

    def __init__(self, classifiers, X, y):
        """Evaluate the performance of the model
        Args:
            classifiers (:obj: `list`): list of classifiers which support fit and transform
            X (np.array): N x M input matrix or data to fit
            y (np.array): N x 1 target class to predict

        """
        self.classifiers = classifiers
        self.X = X
        self.y = y
        self.scoring = ['balanced_accuracy', 'precision', 'recall', 'f1']
        self.sss = StratifiedShuffleSplit(random_state=None)
        self.cv_scores = dict()

    def evaluate_performance(self, n_splits=5, test_size=.2):
        self.sss.n_splits = n_splits
        self.sss.test_size = test_size

        for clf in self.classifiers:
            clf_name = clf.__class__.__name__
            print("*"*80 + "\n{}".format(clf_name))
            scores = cross_validate(
                clf, self.X, self.y, scoring=self.scoring, cv=self.sss, n_jobs=-1)
            self.cv_scores[clf_name] = scores
            self._print_average_performance(scores)

    def _print_average_performance(self, scores):
        for k, v in scores.items():
            print("{:<25}: {:>4.3f}".format(k, np.average(v)))
