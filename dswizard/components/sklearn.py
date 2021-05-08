import numpy as np
from sklearn.base import BaseEstimator

from dswizard.components.base import PredictionMixin


class StackingEstimator(BaseEstimator, PredictionMixin):
    """StackingEstimator

    A shallow wrapper around a classification algorithm to implement the transform method. Allows stacking of
    arbitrary classification algorithms in a pipelines.

    Read more in the :ref:`User Guide`.

    Parameters
    ----------
    estimator : PredictionMixin
        An instance implementing PredictionMixin

    Attributes
    ----------
    estimator : PredictionMixin
        The wrapped PredicitionMixin instance

    See also
    --------
    PredicitionMixin
        Instances that can be stacked using this class.

    References
    ----------
    .. [1] `Stacked Generalization
        <https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.133.8090&rep=rep1&type=pdf>`_
    """

    def __init__(self, estimator: PredictionMixin):
        self.estimator = estimator

    def fit(self, *args):
        self.estimator.fit(*args)
        return self

    def transform(self, X, *args) -> np.ndarray:
        # add class probabilities as a synthetic feature
        # noinspection PyUnresolvedReferences
        try:
            X_transformed = np.hstack((X, self.estimator.predict_proba(X)))
        except AttributeError:
            X_transformed = X

        # add class prediction as a synthetic feature
        # noinspection PyUnresolvedReferences
        X_transformed = np.hstack((X_transformed, np.reshape(self.estimator.predict(X), (-1, 1))))

        return X_transformed

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)

    def get_params(self, deep=True):
        return {'estimator': self.estimator}

    def set_params(self, **params):
        self.estimator = params['estimator']

    def __repr__(self, N_CHAR_MAX=700):
        return f'Stacking({self.estimator.__repr__()})'
