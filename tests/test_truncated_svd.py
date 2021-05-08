import numpy as np
import sklearn

from dswizard.components.feature_preprocessing.truncated_svd import TruncatedSVDComponent
from dswizard.components.util import resolve_factor
from tests import base_test


class TestTruncatedSVDComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = TruncatedSVDComponent(random_state=42)
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.decomposition.TruncatedSVD(random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = TruncatedSVDComponent(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        # Fix n_components from percentage to absolute
        config['n_components'] = min(resolve_factor(config['n_components_factor'], X_train.shape[1]),
                                     (X_train.shape[1] - 1))
        del config['n_components_factor']

        expected = sklearn.decomposition.TruncatedSVD(**config, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)
