import numpy as np
import sklearn

from dswizard.components.feature_preprocessing.random_trees_embedding import RandomTreesEmbeddingComponent
from dswizard.components.util import resolve_factor
from tests import base_test


class TestRandomTreesEmbeddingComponent(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = RandomTreesEmbeddingComponent(random_state=42)
        config: dict = self.get_default(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = sklearn.ensemble.RandomTreesEmbedding(random_state=42, n_jobs=1, sparse_output=False)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = RandomTreesEmbeddingComponent(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        config['max_depth'] = max(resolve_factor(config['max_depth_factor'], X_train.shape[1]), 2)
        del config['max_depth_factor']

        config['max_leaf_nodes'] = max(resolve_factor(config['max_leaf_nodes_factor'], X_train.shape[0]), 2)
        del config['max_leaf_nodes_factor']

        config['min_samples_leaf'] = resolve_factor(config['min_samples_leaf_factor'], X_train.shape[0])
        del config['min_samples_leaf_factor']

        config['min_samples_split'] = max(resolve_factor(config['min_samples_split_factor'], X_train.shape[0]), 2)
        del config['min_samples_split_factor']

        expected = sklearn.ensemble.RandomTreesEmbedding(**config, n_jobs=1, sparse_output=False, random_state=42)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert repr(actual.estimator) == repr(expected)
        assert np.allclose(X_actual, X_expected)
