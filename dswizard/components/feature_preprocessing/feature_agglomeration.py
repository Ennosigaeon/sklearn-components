import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenInClause, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, \
    HANDLES_NUMERIC, HANDLES_MULTICLASS, resolve_factor


class FeatureAgglomerationComponent(PreprocessingAlgorithm):
    def __init__(self, n_clusters_factor: int = 2,
                 affinity: str = "euclidean",
                 compute_full_tree: str = "auto",
                 linkage: str = "ward",
                 pooling_func: str = "mean",
                 distance_threshold: float = None):
        super().__init__()
        self.n_clusters_factor = n_clusters_factor
        self.affinity = affinity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.distance_threshold = distance_threshold
        self.pooling_func = pooling_func

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.cluster import FeatureAgglomeration

        if self.pooling_func == "mean":
            pooling_func = np.mean
        elif self.pooling_func == "median":
            pooling_func = np.median
        elif self.pooling_func == "max":
            pooling_func = np.max
        else:
            raise ValueError(f'Unknown pooling function \'{self.pooling_func}\'')

        if self.distance_threshold is not None:
            n_clusters = None
            self.compute_full_tree = True
        else:
            if isinstance(self.n_clusters_factor, int):
                n_clusters = self.n_clusters_factor
            else:
                n_clusters = max(min(resolve_factor(self.n_clusters_factor, n_features, default=2, cs_default=1.),
                                     (n_features - 1)), 2)

        return FeatureAgglomeration(n_clusters=n_clusters,
                                    affinity=self.affinity,
                                    compute_full_tree=self.compute_full_tree,
                                    linkage=self.linkage,
                                    distance_threshold=self.distance_threshold,
                                    pooling_func=pooling_func)

    @staticmethod
    def get_properties():
        return {'shortname': 'FA',
                'name': 'Feature Agglomeration',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        n_clusters_factor = UniformFloatHyperparameter("n_clusters_factor", 0., 1., default_value=1.)
        affinity = CategoricalHyperparameter("affinity", ["euclidean", "manhattan", "cosine"],
                                             default_value="euclidean")
        linkage = CategoricalHyperparameter("linkage", ["ward", "complete", "average"], default_value="ward")
        pooling_func = CategoricalHyperparameter("pooling_func", ["mean", "median", "max"], default_value="mean")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_clusters_factor, affinity, linkage, pooling_func])

        affinity_and_linkage = ForbiddenAndConjunction(ForbiddenInClause(affinity, ["manhattan", "cosine"]),
                                                       ForbiddenEqualsClause(linkage, "ward"))
        cs.add_forbidden_clause(affinity_and_linkage)
        return cs
