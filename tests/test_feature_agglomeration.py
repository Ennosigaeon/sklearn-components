import numpy as np
import sklearn

from dswizard.components.feature_preprocessing.feature_agglomeration import FeatureAgglomerationComponent
from dswizard.components.util import resolve_factor
from tests import base_test


class TestFeatureAgglomerationComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = FeatureAgglomerationComponent()
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.cluster.FeatureAgglomeration()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = FeatureAgglomerationComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        if config['pooling_func'] == "mean":
            config['pooling_func'] = np.mean
        elif config['pooling_func'] == "median":
            config['pooling_func'] = np.median
        elif config['pooling_func'] == "max":
            config['pooling_func'] = np.max

        config['n_clusters'] = max(
            min(resolve_factor(config['n_clusters_factor'], X_train.shape[1]), (X_train.shape[1] - 1)), 2)
        del config['n_clusters_factor']

        expected = sklearn.cluster.FeatureAgglomeration(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)
