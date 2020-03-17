from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition
from scipy import sparse

from automl.components.base import PreprocessingAlgorithm


class TruncatedSVDComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_components: int = 2,
                 algorithm: str = 'randomized',
                 n_iter: int = 5,
                 tol: float = None,
                 random_state: int = None):
        super().__init__()
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        from sklearn.decomposition import TruncatedSVD
        self.preprocessor = TruncatedSVD(n_components=n_components,
                                         algorithm=algorithm,
                                         n_iter=n_iter,
                                         tol=tol)

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
        n_components = UniformIntegerHyperparameter(name="n_components", lower=1, upper=20, default_value=2)
        n_iter = UniformIntegerHyperparameter(name="n_iter", lower=1, upper=20, default_value=5)
        tol = UniformFloatHyperparameter(name="tol", lower=1e-5, upper=10, default_value=0.01)
        algorithm = CategoricalHyperparameter(name="algorithm", choices=["arpack", "randomized"],
                                              default_value="randomized")

        cs = ConfigurationSpace()
        cs.add_hyperparameters([n_components, algorithm, n_iter, tol])
        tol_condition = InCondition(tol, algorithm, ["arpack"])
        cs.add_conditions([tol_condition])

        return cs