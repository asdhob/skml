import random
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, ClassifierMixin
from sklearn.utils import validation
from random import shuffle
from random import random
from random import sample

from ..problem_transformation import ClassifierChain


class EnsembleClassifierChain(
        BaseEstimator, MetaEstimatorMixin, ClassifierMixin):
    def __init__(
            self,
            estimator,
            number_of_chains=10,
            threshold=.5,
            max_features=0.67):
        """
        Ensemble of classifier chains (ECC) trains an ensemble of bagged
        classifier chains. Each chain is trained on a randomly sampled subset
        of the training data (with replacement, also known as bagging).

        Parameters
        ----------
        estimator : scikit-learn compatible classifier instance.
            Will be copied (with all hyperparameters) before use,
            hence will be left untouched.
        number_of_chains : number, default = 10
            Number of chains the ensemble shall train
        threshold : number in [0,1], default = 0.5
            Decision threshold to assign a label or not. Has to be
            between 0 and 1.
        max_features : number in [0, 1], default = 1.0
            Fractions of features to use at once.

        Returns
        -------
        ensemble classifier chain instance
        """
        self.number_of_chains = number_of_chains
        self.estimator = estimator
        self.threshold = threshold
        self.max_features = max_features
        self.estimators_ = []

    def fit(self, X, y):
        """
        Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-label targets.
        """

        validation.check_X_y(X, y, multi_output=True)
        y = validation.check_array(y, accept_sparse=True)

        for i in range(self.number_of_chains):
            # the classifier gets cloned internally in classifer chains, so
            # no need to do that here.
            cc = ClassifierChain(self.estimator)

            no_samples = y.shape[0]
            no_cols = y.shape[1]
        
#            shuffled_col_id = np.array(shuffle([i for i in range(no_cols)]))
            my_list = range(no_cols)
            shuffled_col_id = list(sorted(my_list, key=lambda x: random()))
            print(type(shuffled_col_id))
            print(shuffled_col_id)
        
            # create random subset for each chain individually
            idx = sample(range(no_samples),
                                int(no_samples * self.max_features))
            print(type(idx))
            print(idx)        
            cc.fit(X[idx, :], y[np.array(idx)[:, None].astype(int), shuffled_col_id])

            self.estimators_.append(cc)

    def predict(self, X):
        """
        Predicts the labels for the given instances.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        array-like, shape = [n_samples, n_labels]
            Estimated labels
        """
        validation.check_is_fitted(self, 'estimators_')
        
        # define function for getting max/min freq for 1/0 class
        def get_best_prob(x):
                if np.mean(x)>=0.5:
                        return max(x[x>=self.threshold])
                else:
                        return min(x[x<self.threshold])

        preds = np.array([cc.predict(X) for cc in self.estimators_])
        #print(preds)
        preds = np.apply_along_axis(func1d=get_best_prob, axis=0,arr=preds)
        preds = np.mean(preds, axis=0)
        W_norm = preds.mean(axis=0)
        out = preds / W_norm
        
        return (out >= self.threshold).astype(int)

#        return preds

