from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.components.util import HANDLES_NOMINAL_CLASS, HANDLES_MISSING, HANDLES_NOMINAL, HANDLES_NUMERIC, \
    HANDLES_MULTICLASS, resolve_factor


class SelectKBestComponent(PreprocessingAlgorithm):
    def __init__(self,
                 score_func: str = "f_classif",
                 k_factor: float = 0.5):
        super().__init__()
        self.score_func = score_func
        self.k_factor = k_factor

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, f_regression
        if self.score_func == "chi2":
            score_func = chi2
        elif self.score_func == "f_classif":
            score_func = f_classif
        elif self.score_func == "mutual_info":
            score_func = mutual_info_classif
        elif self.score_func == "f_regression":
            score_func = f_regression
        else:
            raise ValueError("score_func must be in ('chi2, 'f_classif', 'mutual_info'), but is: %s" % self.score_func)

        from sklearn.feature_selection import SelectKBest
        k = resolve_factor(self.k_factor, n_features)
        return SelectKBest(score_func=score_func, k=k)

    def transform(self, X):
        import scipy.sparse
        import sklearn.feature_selection

        # TODO really? I assume only copied from auto-sklearn
        # Because the pipeline guarantees that each feature is positive,
        # clip all values below zero to zero
        if self.score_func == sklearn.feature_selection.chi2:
            if scipy.sparse.issparse(X):
                X.data[X.data < 0] = 0.0
            else:
                X[X < 0] = 0.0

        if self.estimator is None:
            raise NotImplementedError()
        Xt = self.estimator.transform(X)
        if Xt.shape[1] == 0:
            raise ValueError("%s removed all features." % self.__class__.__name__)
        return Xt

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()

        k_factor = UniformFloatHyperparameter("k_factor", 0., 1., default_value=0.5)
        score_func = CategoricalHyperparameter(name="score_func",
                                               choices=["chi2", "f_classif", "mutual_info", "f_regression"],
                                               default_value="f_classif")

        cs.add_hyperparameters([score_func, k_factor])
        return cs

    @staticmethod
    def get_properties():
        return {'shortname': 'FastICA',
                'name': 'Fast Independent Component Analysis',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True}
