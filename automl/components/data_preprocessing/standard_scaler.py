from automl.components.base import PreprocessingAlgorithm
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, UniformIntegerHyperparameter

class StandardScalerComponent(PreprocessingAlgorithm):

    def __init__(self, with_mean: bool = True, with_std: bool = True, copy: bool = True):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        from sklearn.preprocessing import StandardScaler
        self.preprocessor = StandardScaler(copy=self.copy, with_std=self.with_std, with_mean=self.with_mean)

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        with_mean = CategoricalHyperparameter("with_mean", [True,False], default_value=True)
        with_std = CategoricalHyperparameter("with_std", [True,False], default_value=True)
        copy = CategoricalHyperparameter("copy", [True,False], default_value=True)

        cs.add_hyperparameters([with_mean, with_std, copy])
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'StandardScaler',
                'name': 'StandardScaler',
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
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT, SIGNED_DATA),
                'preferred_dtype': None}
