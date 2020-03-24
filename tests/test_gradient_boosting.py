import numpy as np
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier

from automl.components.classification.gradient_boosting import GradientBoostingClassifier
from automl.util.common import resolve_factor
from tests import base_test


class TestGradientBoosting(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GradientBoostingClassifier(random_state=42)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = HistGradientBoostingClassifier(random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data(multiclass=False)

        actual = GradientBoostingClassifier(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        config['max_depth'] = resolve_factor(config['max_depth_factor'], X_train.shape[1])
        del config['max_depth_factor']

        config['max_leaf_nodes'] = resolve_factor(config['max_leaf_nodes_factor'], X_train.shape[0])
        del config['max_leaf_nodes_factor']

        expected = HistGradientBoostingClassifier(**config, random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(y_actual, y_expected)
