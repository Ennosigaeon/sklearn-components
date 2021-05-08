import numpy as np
import sklearn

from dswizard.components.feature_preprocessing.select_k_best import SelectKBestComponent
from dswizard.components.util import resolve_factor
from tests import base_test


class TestSelectKBestComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = SelectKBestComponent(k_factor=0.5)
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.feature_selection.SelectKBest(k=2)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = SelectKBestComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        if config['score_func'] == "chi2":
            config['score_func'] = sklearn.feature_selection.chi2
        elif config['score_func'] == "f_classif":
            config['score_func'] = sklearn.feature_selection.f_classif
        elif config['score_func'] == "mutual_info":
            config['score_func'] = sklearn.feature_selection.mutual_info_classif
        elif config['score_func'] == "f_regression":
            config['score_func'] = sklearn.feature_selection.f_regression

        config['k'] = resolve_factor(config['k_factor'], X_train.shape[1])
        del config['k_factor']

        expected = sklearn.feature_selection.SelectKBest(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)
