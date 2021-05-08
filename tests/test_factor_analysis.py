import numpy as np
import sklearn

from dswizard.components.feature_preprocessing.factor_analysis import FactorAnalysisComponent
from dswizard.components.util import resolve_factor
from tests import base_test


class TestFactorAnalysisComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = FactorAnalysisComponent(random_state=42)
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(np.copy(X_train), np.copy(y_train))
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.decomposition.FactorAnalysis(n_components=40, random_state=42, copy=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = FactorAnalysisComponent(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(np.copy(X_train), np.copy(y_train))
        X_actual = actual.transform(np.copy(X_test))

        config['n_components'] = resolve_factor(config['n_components_factor'], X_train.shape[1])
        del config['n_components_factor']

        expected = sklearn.decomposition.FactorAnalysis(**config, random_state=42, copy=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)
