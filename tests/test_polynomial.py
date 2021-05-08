import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from dswizard.components.feature_preprocessing.polynomial import PolynomialFeaturesComponent
from tests import base_test


class TestPolynomial(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = PolynomialFeaturesComponent()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = PolynomialFeatures()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = PolynomialFeaturesComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = PolynomialFeatures(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)
