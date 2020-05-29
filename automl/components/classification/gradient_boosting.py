from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    Constant, CategoricalHyperparameter

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none, HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS
from automl.util.common import resolve_factor


# TODO does not honour affinity restrictions

class GradientBoostingClassifier(PredictionAlgorithm):
    def __init__(self,
                 loss: str = 'auto',
                 learning_rate: float = 0.1,
                 max_iter: int = 100,
                 min_samples_leaf: int = 20,
                 max_depth_factor: int = None,
                 max_leaf_nodes_factor: int = 31,
                 max_bins: int = 255,
                 l2_regularization: float = 0.,
                 tol: float = 1e-7,
                 scoring: str = 'f1_weighted',
                 n_iter_no_change: int = None,
                 validation_fraction: float = 0.1,
                 random_state=None):
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.max_depth_factor = max_depth_factor
        self.max_leaf_nodes_factor = max_leaf_nodes_factor
        self.max_bins = max_bins
        self.l2_regularization = l2_regularization
        self.tol = tol
        self.scoring = scoring
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier

        if check_none(self.scoring):
            self.scoring = None

        # Heuristic to set the tree depth
        max_depth = resolve_factor(self.max_depth_factor, n_features, cs_default=1.)
        if max_depth is not None:
            max_depth = max(max_depth, 2)

        l2_regularization = 0. if self.l2_regularization == 1e-07 else self.l2_regularization

        # Heuristic to set the tree width
        if isinstance(self.max_leaf_nodes_factor, int):
            max_leaf_nodes = self.max_leaf_nodes_factor
        else:
            max_leaf_nodes = resolve_factor(self.max_leaf_nodes_factor, n_samples, default=31, cs_default=1.)
        if max_leaf_nodes is not None:
            max_leaf_nodes = max(max_leaf_nodes, 2)

        # Heuristic to set the tree width
        if isinstance(self.min_samples_leaf, int):
            min_samples_leaf = self.min_samples_leaf
        else:
            min_samples_leaf = resolve_factor(self.min_samples_leaf, n_samples, default=20, cs_default=0.0001)

        n_iter_no_change = None if self.n_iter_no_change == 0 else self.n_iter_no_change

        if self.scoring == 'balanced_accurary':
            self.scoring = 'balanced_accuracy'

        return HistGradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            max_bins=self.max_bins,
            l2_regularization=l2_regularization,
            tol=self.tol,
            scoring=self.scoring,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
        )

    @staticmethod
    def get_properties():
        return {'shortname': 'GB',
                'name': 'Gradient Boosting Classifier',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True
                }

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        loss = Constant("loss", "auto")
        learning_rate = UniformFloatHyperparameter(name="learning_rate", lower=1e-6, upper=1.5, default_value=0.1,
                                                   log=True)
        max_depth_factor = UniformFloatHyperparameter("max_depth_factor", 1e-5, 2.5, default_value=1.)
        max_iter = UniformIntegerHyperparameter("max_iter", 0, 1000, default_value=100)
        max_leaf_nodes_factor = UniformFloatHyperparameter("max_leaf_nodes_factor", 1e-5, 1., default_value=1.)
        min_samples_leaf = UniformFloatHyperparameter("min_samples_leaf", 0.0001, 0.5, default_value=0.0001)
        l2_regularization = UniformFloatHyperparameter(name="l2_regularization", lower=1e-7, upper=10.,
                                                       default_value=1e-7, log=True)
        max_bins = UniformIntegerHyperparameter("max_bins", 5, 255, default_value=255)
        tol = UniformFloatHyperparameter("tol", 0., 0.25, default_value=1e-7)
        scoring = CategoricalHyperparameter("scoring",
                                            ["accuracy", "balanced_accuracy", "balanced_accurary", "average_precision",
                                             "neg_brier_score",
                                             "f1", "f1_micro", "f1_macro", "f1_weighted", "f1_samples", "neg_log_loss",
                                             "precision", "precision_micro", "precision_macro", "precision_weighted",
                                             "precision_samples", "recall", "recall_micro", "recall_macro",
                                             "recall_weighted", "recall_samples", "jaccard", "jaccard_micro",
                                             "jaccard_macro", "jaccard_weighted", "jaccard_samples", "roc_auc",
                                             "roc_auc_ovr", "roc_auc_ovo", "roc_auc_ovr_weighted",
                                             "roc_auc_ovo_weighted"], default_value="f1_weighted")
        n_iter_no_change = UniformIntegerHyperparameter(name="n_iter_no_change", lower=0, upper=100, default_value=0)
        validation_fraction = UniformFloatHyperparameter(name="validation_fraction", lower=0.001, upper=1.0,
                                                         default_value=0.1)

        cs.add_hyperparameters([loss, learning_rate, max_iter, min_samples_leaf, max_leaf_nodes_factor, max_bins,
                                l2_regularization, tol, scoring, n_iter_no_change, validation_fraction,
                                max_depth_factor])

        return cs
