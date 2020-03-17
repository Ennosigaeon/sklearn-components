from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    UnParametrizedHyperparameter, Constant, CategoricalHyperparameter

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none


class AdaBoostingClassifier(PredictionAlgorithm):
    def __init__(self,
                 algorithm: str = 'SAMME.R',
                 learning_rate: float = 1.0,
                 n_estimators: int = 50,
                 random_state=None,
                 ):
        super().__init__()
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, Y):
        from sklearn.ensemble import AdaBoostClassifier

        self.estimator = AdaBoostClassifier(
            algorithm=self.algorithm,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )

        self.estimator.fit(X, Y)
        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'AB',
                'name': 'Ada Boosting Classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (PREDICTIONS,)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        learning_rate = UniformFloatHyperparameter(name="learning_rate", lower=1, upper=2, default_value=1.0,
                                                   log=True)
        n_estimators = UniformIntegerHyperparameter("n_estimators", 25, 150, default_value=50)
        algorithm = CategoricalHyperparameter("algorithm", ["SAMME", "SAMME.R"], default_value="SAMME")

        cs.add_hyperparameters(
            [learning_rate, n_estimators, algorithm])

        return cs