from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

from automl.components.base import PredictionAlgorithm
from automl.util.common import resolve_factor
from automl.util.util import convert_multioutput_multiclass_to_multilabel


class DecisionTree(PredictionAlgorithm):

    def __init__(self,
                 criterion: str = "gini",
                 splitter: str = "best",
                 max_depth_factor: float = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.,
                 max_features: float = None,
                 random_state=None,
                 max_leaf_nodes_factor: int = None,
                 min_impurity_decrease: float = 0.,
                 class_weight=None,
                 ccp_alpha: float = 0.
                 ):
        super().__init__()
        self.criterion = criterion
        self.splitter = splitter
        self.max_features = max_features
        self.max_depth_factor = max_depth_factor
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes_factor = max_leaf_nodes_factor
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None):
        self.estimator = self.to_sklearn(X.shape[0], X.shape[1])
        self.estimator.fit(X, y, sample_weight=sample_weight)
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.tree import DecisionTreeClassifier

        # Heuristic to set the tree depth
        max_depth = resolve_factor(self.max_depth_factor, n_features)
        if max_depth is not None:
            max_depth = max(max_depth, 2)

        # Heuristic to set the tree width
        max_leaf_nodes = resolve_factor(self.max_leaf_nodes_factor, n_samples)
        if max_leaf_nodes is not None:
            max_leaf_nodes = max(max_leaf_nodes, 2)

        return DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_impurity_decrease=self.min_impurity_decrease,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            random_state=self.random_state)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'DT',
                'name': 'Decision Tree Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (PREDICTIONS,)}
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        criterion = CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini")
        splitter = CategoricalHyperparameter("splitter", ["best", "random"], default_value="best")
        max_depth_factor = UniformFloatHyperparameter("max_depth_factor", 1e-7, 2.5, default_value=1.)
        min_samples_split = UniformFloatHyperparameter("min_samples_split", 1e-7, 0.5, default_value=0.0001)
        min_samples_leaf = UniformFloatHyperparameter("min_samples_leaf", 1e-7, 0.5, default_value=0.0001)
        min_weight_fraction_leaf = UniformFloatHyperparameter("min_weight_fraction_leaf", 0., 0.5, default_value=0.)
        max_features = UniformFloatHyperparameter('max_features', 1e-4, 1., default_value=1.)
        max_leaf_nodes_factor = UniformFloatHyperparameter("max_leaf_nodes_factor", 1e-7, 1., default_value=1.)
        min_impurity_decrease = UniformFloatHyperparameter('min_impurity_decrease', 0., 1., default_value=0.)
        ccp_alpha = UniformFloatHyperparameter("ccp_alpha", 0., 1., default_value=0.)

        cs.add_hyperparameters([criterion, splitter, max_features, max_depth_factor,
                                min_samples_split, min_samples_leaf,
                                min_weight_fraction_leaf, max_leaf_nodes_factor,
                                min_impurity_decrease, ccp_alpha])

        return cs
