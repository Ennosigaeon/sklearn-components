from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

from scipy import sparse

from automl.components.base import PreprocessingAlgorithm


class TruncatedSVDComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_components: float = 0.5,
                 algorithm: str = 'randomized',
                 n_iter: int = 5,
                 tol: float = 0.,
                 random_state: int = None):
        super().__init__()
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, Y):
        import numpy as np
        num_features = X.shape[1]
        self.n_components = max(1, int(np.round(self.n_components * num_features, 0)))
        from sklearn.decomposition import TruncatedSVD
        self.preprocessor = TruncatedSVD(n_components=self.n_components,
                                         algorithm=self.algorithm,
                                         n_iter=self.n_iter,
                                         tol=self.tol,
                                         random_state=self.random_state).fit(X, Y)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'TSVD',
                'name': 'Truncated Singular Value Decomposition',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (SPARSE, UNSIGNED_DATA),
                # 'output': (DENSE, INPUT)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        n_components = UniformFloatHyperparameter(name="n_components", lower=0., upper=1., default_value=0.5)
        n_iter = UniformIntegerHyperparameter(name="n_iter", lower=1, upper=100, default_value=5)
        tol = UniformFloatHyperparameter(name="tol", lower=1e-7, upper=5., default_value=0.01)
        algorithm = CategoricalHyperparameter(name="algorithm", choices=["arpack", "randomized"],
                                              default_value="randomized")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_components, algorithm, n_iter, tol])
        tol_condition = InCondition(tol, algorithm, ["arpack"])
        cs.add_conditions([tol_condition])

        return cs