from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant

from automl.components.base import PreprocessingAlgorithm


class BinarizerComponent(PreprocessingAlgorithm):

    def __init__(self, threshold: float = 0.):
        super().__init__()
        self.threshold = threshold

    def fit(self, X, y=None):
        from sklearn.preprocessing import Binarizer
        self.preprocessor = Binarizer(threshold=self.threshold)
        self.preprocessor = self.preprocessor.fit(X)
        return self

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        # TODO Use fraction of data per column
        threshold = Constant('threshold', 0.)
        cs.add_hyperparameter(threshold)
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        # TODO find out of this is right!
        return {'shortname': 'Binarizer',
                'name': 'Binarizer',
                'handles_missing_values': False,
                'handles_nominal_values': False,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                'handles_sparse': True,
                'handles_dense': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT, SIGNED_DATA),
                'preferred_dtype': None}