from typing import Dict, Type

__author__ = 'feurerm'

import os
from collections import OrderedDict

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from dswizard.components.base import PredictionAlgorithm, find_components, ComponentChoice, PredictionMixin, \
    EstimatorComponent

classifier_directory = os.path.split(__file__)[0]
_classifiers = find_components(__package__, classifier_directory, PredictionAlgorithm)


class ClassifierChoice(ComponentChoice, PredictionMixin):

    def get_components(self) -> Dict[str, Type[EstimatorComponent]]:
        components = OrderedDict()
        components.update(_classifiers)
        return components

    def get_hyperparameter_search_space(self, mf=None,
                                        default=None,
                                        include=None,
                                        exclude=None,
                                        **kwargs):

        if include is not None and exclude is not None:
            raise ValueError("The arguments include_estimators and "
                             "exclude_estimators cannot be used together.")

        cs = ConfigurationSpace()

        # Compile a list of all estimator objects for this problem
        available_estimators = self.get_available_components(mf=mf, include=include, exclude=exclude)

        if len(available_estimators) == 0:
            raise ValueError("No classifiers found")

        if default is None:
            defaults = ['random_forest', 'liblinear_svc', 'sgd',
                        'libsvm_svc'] + list(available_estimators.keys())
            for default_ in defaults:
                if default_ in available_estimators:
                    if include is not None and default_ not in include:
                        continue
                    if exclude is not None and default_ in exclude:
                        continue
                    default = default_
                    break

        estimator = CategoricalHyperparameter('__choice__', list(available_estimators.keys()), default_value=default)
        cs.add_hyperparameter(estimator)
        for estimator_name in available_estimators.keys():
            estimator_configuration_space = available_estimators[estimator_name].get_hyperparameter_search_space()
            parent_hyperparameter = {'parent': estimator, 'value': estimator_name}
            cs.add_configuration_space(estimator_name, estimator_configuration_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space_ = cs
        return cs

    def fit(self, X, y, **kwargs):
        if kwargs is None:
            kwargs = {}
        return self.estimator.fit(X, y, **kwargs)

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def estimator_supports_iterative_fit(self):
        return hasattr(self.estimator, 'iterative_fit')

    def iterative_fit(self, X, y, n_iter=1, **fit_params):
        if fit_params is None:
            fit_params = {}
        return self.estimator.iterative_fit(X, y, n_iter=n_iter, **fit_params)

    def configuration_fully_fitted(self):
        return self.estimator.configuration_fully_fitted()
