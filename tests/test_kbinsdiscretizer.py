import numpy as np
import sklearn
from dswizard.components.feature_preprocessing.kbinsdiscretizer import KBinsDiscretizer
from scipy.sparse import csr_matrix

from tests import base_test


class TestKBinsDiscretizer(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = KBinsDiscretizer()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.preprocessing.KBinsDiscretizer()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)
        if isinstance(X_expected, csr_matrix):
            X_expected = X_expected.todense()

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = KBinsDiscretizer()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.preprocessing.KBinsDiscretizer(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)
        if isinstance(X_expected, csr_matrix):
            X_expected = X_expected.todense()

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)
