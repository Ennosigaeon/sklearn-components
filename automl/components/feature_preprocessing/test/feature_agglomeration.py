from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenInClause, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter

from automl.components.base import PreprocessingAlgorithm
from automl.util.common import check_for_bool


class FeatureAgglomerationComponent(PreprocessingAlgorithm):
    def __init__(self, n_clusters: int = 2,
                 affinity: str = "euclidean",
                 compute_full_tree: str = "auto",
                 linkage: str = "ward",
                 distance_threshold: float = None):
        super().__init__()
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.distance_threshold = distance_threshold

    def fit(self, X, y=None):
        from sklearn.cluster import FeatureAgglomeration

        if self.distance_threshold is not None:
            self.n_clusters = None
            self.compute_full_tree = True

        self.preprocessor = FeatureAgglomeration(n_clusters=self.n_clusters,
                                                 affinity=self.affinity,
                                                 compute_full_tree=self.compute_full_tree,
                                                 linkage=self.linkage,
                                                 distance_threshold=self.distance_threshold)
        self.preprocessor.fit(X, y)
        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'FA',
                'name': 'Feature Agglomeration',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT,)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 10, default_value=2)
        affinity = CategoricalHyperparameter("affinity",
                                             ["euclidean", "l1", "l2", "manhattan", "cosine", "precomputed"],
                                             default_value="euclidean")
        compute_full_tree = CategoricalHyperparameter("compute_full_tree", [True, False, 'auto'], default_value="auto")
        linkage = CategoricalHyperparameter("linkage", ["ward", "complete", "average", "single"], default_value="ward")
        distance_threshold = UniformFloatHyperparameter("distance_threshold", 0.001, 0.5, default_value=None)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_clusters, affinity, compute_full_tree, linkage, distance_threshold])

        affinity_and_linkage = ForbiddenAndConjunction(
            ForbiddenInClause(affinity, ["manhattan", "cosine"]),
            ForbiddenEqualsClause(linkage, "ward"))
        cs.add_forbidden_clause(affinity_and_linkage)

        return cs