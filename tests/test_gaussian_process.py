import numpy as np
import sklearn.naive_bayes
import sklearn.svm

from automl.components.classification.testg.gaussian_process import GaussianProcessClassifier
from tests import base_test


class TestGaussianProcessClassifier(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GaussianProcessClassifier(random_state=42)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.gaussian_process.GaussianProcessClassifier(random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert np.allclose(y_actual, y_expected)
        assert repr(actual.estimator) == repr(expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GaussianProcessClassifier(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.gaussian_process.GaussianProcessClassifier(**config, random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert np.allclose(y_actual, y_expected)
        assert repr(actual.estimator) == repr(expected)