import os
from collections import OrderedDict
from typing import Dict, Type

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from dswizard.components.base import PreprocessingAlgorithm, find_components, ComponentChoice, EstimatorComponent

preprocessor_directory = os.path.split(__file__)[0]
_preprocessors = find_components(__package__,
                                 preprocessor_directory,
                                 PreprocessingAlgorithm)


class DataPreprocessorChoice(ComponentChoice):

    def get_components(self) -> Dict[str, Type[EstimatorComponent]]:
        components = OrderedDict()
        components.update(_preprocessors)
        return components

    def get_hyperparameter_search_space(self, mf=None,
                                        default=None,
                                        include=None,
                                        exclude=None,
                                        **kwargs):
        cs = ConfigurationSpace()

        # Compile a list of legal preprocessors for this problem
        available_preprocessors = self.get_available_components(mf=mf, include=include, exclude=exclude)

        if len(available_preprocessors) == 0:
            raise ValueError("No preprocessors found, please add NoPreprocessing")

        if default is None:
            defaults = ['no_preprocessing', 'select_percentile', 'pca', 'truncatedSVD']
            for default_ in defaults:
                if default_ in available_preprocessors:
                    default = default_
                    break

        preprocessor = CategoricalHyperparameter('__choice__', list(available_preprocessors.keys()),
                                                 default_value=default)
        cs.add_hyperparameter(preprocessor)
        for name in available_preprocessors:
            preprocessor_configuration_space = available_preprocessors[name].get_hyperparameter_search_space()
            parent_hyperparameter = {'parent': preprocessor, 'value': name}
            cs.add_configuration_space(name, preprocessor_configuration_space,
                                       parent_hyperparameter=parent_hyperparameter)

        self.configuration_space_ = cs
        return cs

    def transform(self, X):
        return self.estimator.transform(X)

    def fit_transform(self, X, y=None):
        return self.estimator.fit(X, y).transform(X)
