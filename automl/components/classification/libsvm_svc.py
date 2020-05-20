from ConfigSpace import ForbiddenAndConjunction, ForbiddenInClause
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from automl.components.base import PredictionAlgorithm
from automl.util.common import check_none, check_for_bool, HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, \
    HANDLES_MISSING, HANDLES_NOMINAL_CLASS
from automl.util.util import softmax


# noinspection PyPep8Naming
class LibSVM_SVC(PredictionAlgorithm):
    def __init__(self,
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: str = 'scale',
                 coef0: float = 0.0,
                 shrinking: bool = True,
                 tol: float = 1e-3,
                 class_weight=None,
                 random_state=None,
                 decision_function_shape: str = "ovr",
                 break_ties: bool = False,
                 probability: bool = False
                 ):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.tol = tol
        self.class_weight = class_weight
        self.random_state = random_state
        self.decision_function_shape = decision_function_shape
        self.probability = probability
        self.break_ties = break_ties

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, **kwargs):
        from sklearn.svm import SVC

        if self.degree is None:
            self.degree = 3
        if self.gamma is None:
            self.gamma = 0.0
        if self.coef0 is None:
            self.coef0 = 0.0

        self.shrinking = check_for_bool(self.shrinking)

        if check_none(self.class_weight):
            self.class_weight = None

        return SVC(C=self.C,
                   kernel=self.kernel,
                   degree=self.degree,
                   gamma=self.gamma,
                   coef0=self.coef0,
                   shrinking=self.shrinking,
                   tol=self.tol,
                   class_weight=self.class_weight,
                   random_state=self.random_state,
                   cache_size=200,
                   probability=self.probability,
                   break_ties=self.break_ties,
                   decision_function_shape=self.decision_function_shape
                   )

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        decision = self.estimator.decision_function(X)
        return softmax(decision)

    @staticmethod
    def get_properties():
        return {'shortname': 'LibSVM-SVC',
                'name': 'LibSVM Support Vector Classification',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True
                }

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        C = UniformFloatHyperparameter("C", 1e-7, 1e5, default_value=1.0, log=True)
        # No linear kernel here, because we have liblinear
        kernel = CategoricalHyperparameter(name="kernel", choices=["linear", "rbf", "poly", "sigmoid"],
                                           default_value="rbf")
        degree = UniformIntegerHyperparameter("degree", 2, 6, default_value=3)
        gamma = UniformFloatHyperparameter("gamma", 1e-7, 1e5, log=True, default_value=0.1)
        coef0 = UniformFloatHyperparameter("coef0", -30, 1e4, default_value=0.)
        # probability is no hyperparameter, but an argument to the SVM algo
        shrinking = CategoricalHyperparameter("shrinking", [True, False], default_value=True)
        probability = CategoricalHyperparameter("probability", [True, False], default_value=False)
        tol = UniformFloatHyperparameter("tol", 1e-7, 0.5, default_value=1e-3, log=True)
        # cache size is not a hyperparameter, but an argument to the program!
        decision_function_shape = CategoricalHyperparameter("decision_function_shape", ["ovr", "ovo"],
                                                            default_value="ovr")
        break_ties = CategoricalHyperparameter("break_ties", [True, False], default_value=False)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking, tol, probability,
                                decision_function_shape, break_ties])

        degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
        coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
        gamma_condition = InCondition(gamma, kernel, ["rbf", "poly", "sigmoid"])
        cs.add_condition(degree_depends_on_poly)
        cs.add_condition(coef0_condition)
        cs.add_condition(gamma_condition)

        funcshapAndBreakties = ForbiddenAndConjunction(
            ForbiddenInClause(decision_function_shape, ["ovo"]),
            ForbiddenInClause(break_ties, [True])
        )
        cs.add_forbidden_clause(funcshapAndBreakties)

        return cs
