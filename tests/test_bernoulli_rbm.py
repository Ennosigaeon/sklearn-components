import numpy as np
import sklearn

from dswizard.components.feature_preprocessing.bernoulli_rbm import BernoulliRBM
from tests import base_test


class TestBernoulliRBM(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = BernoulliRBM(random_state=42)
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.neural_network.BernoulliRBM(random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = BernoulliRBM(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.neural_network.BernoulliRBM(**config, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)
