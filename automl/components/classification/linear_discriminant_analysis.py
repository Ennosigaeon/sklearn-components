from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    CategoricalHyperparameter

from ConfigSpace.conditions import InCondition
from automl.components.base import PredictionAlgorithm
from automl.util.util import convert_multioutput_multiclass_to_multilabel
from automl.util.common import HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, HANDLES_MISSING, \
    HANDLES_NOMINAL_CLASS, resolve_factor


class LinearDiscriminantAnalysis(PredictionAlgorithm):
    def __init__(self,
                 solver: str = 'svd',
                 shrinkage: str = None,
                 tol: float = 1.0e-4,
                 n_components: int = None
                 ):
        super().__init__()
        self.solver = solver
        self.shrinkage = shrinkage
        self.tol = tol
        self.n_components = n_components

    def fit(self, X, Y):
        import numpy as np

        self.estimator = self.to_sklearn(X.shape[0], X.shape[1], n_classes=len(np.unique(Y)))
        self.estimator.fit(X, Y)
        return self

    def to_sklearn(self, n_samples: int = 0, n_features: int = 0, n_classes: int = 0, **kwargs):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        n_components = resolve_factor(self.n_components, min(n_features, n_classes - 1), cs_default=0.5)

        # initial fit of only increment trees
        return LinearDiscriminantAnalysis(solver=self.solver,
                                          shrinkage=self.shrinkage,
                                          tol=self.tol,
                                          n_components=n_components)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        probas = self.estimator.predict_proba(X)
        probas = convert_multioutput_multiclass_to_multilabel(probas)
        return probas

    @staticmethod
    def get_properties():
        return {'shortname': 'LR',
                'name': 'Logistic Regression',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: False,
                HANDLES_MISSING: False,
                HANDLES_NOMINAL_CLASS: True
                }

    @staticmethod
    def get_hyperparameter_search_space(**kwargs):
        cs = ConfigurationSpace()
        solver = CategoricalHyperparameter("solver", ["svd", "lsqr", "eigen"], default_value="svd")
        shrinkage = UniformFloatHyperparameter("shrinkage", 0., 1., default_value=0.1)
        n_components = UniformFloatHyperparameter("n_components", 0.01, 1., default_value=0.5)
        tol = UniformFloatHyperparameter(name="tol", lower=1.0e-7, upper=1., default_value=1.0e-4, log=True)

        cs.add_hyperparameters([shrinkage, solver, tol, n_components])

        shrinkage_condition = InCondition(shrinkage, solver, ["lsqr", "eigen"])
        cs.add_condition(shrinkage_condition)

        return cs
