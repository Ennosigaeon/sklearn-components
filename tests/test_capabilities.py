import itertools
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder

from dswizard.components.classification import ClassifierChoice
from dswizard.components.data_preprocessing import DataPreprocessorChoice
from dswizard.components.feature_preprocessing import FeaturePreprocessorChoice
from dswizard.components.meta_features import MetaFeatureFactory
from dswizard.components.util import HANDLES_MULTICLASS, HANDLES_MISSING, HANDLES_NOMINAL_CLASS, HANDLES_NUMERIC, \
    HANDLES_NOMINAL


class CapabilitiesTest(TestCase):
    # Dataset   Numeric     Nominal     Missing     Multi-Class     Nominal Class
    # ---------------------------------------------------------------------------
    # 1510      yes         no          no          no              no
    # 40975     no          yes         no          yes             no
    # 1480      yes         yes         no          no              no
    # 61        yes         no          no          yes             yes
    # 15        yes         no          yes         no              yes
    # 23381     yes         yes         yes         no              no

    components = ClassifierChoice().get_components()
    components.update(FeaturePreprocessorChoice().get_components())
    components.update(DataPreprocessorChoice().get_components())

    @staticmethod
    def synthetic_df(numeric: bool = True,
                     nominal: bool = True,
                     missing: bool = True,
                     multi_class: bool = True,
                     nominal_class: bool = True
                     ):
        df = {'base': [1, 2, 3, 4, 5, 6, 7, 8]}
        if numeric:
            df['numeric'] = [1, 0.25, 3, 5.1, 0.2, 0.77, 1.25, 2.8]
        if nominal:
            df['nominal'] = ['plane', 'train', 'auto', 'bus', 'bike', 'boat', 'car', 'submarine']
        if numeric and missing:
            df['numeric_missing'] = [5, np.nan, 5.1, np.nan, 1, 0.5, 2.8, np.nan]
        if nominal and missing:
            df['nominal_missing'] = ['plane', 'train', np.nan, 'train', np.nan, 'bike', 'car', np.nan]

        if multi_class:
            if nominal_class:
                y = ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd']
            else:
                y = [1, 1, 2, 2, 3, 3, 4, 4]
        else:
            if nominal_class:
                y = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
            else:
                y = [1, 1, 1, 1, 2, 2, 2, 2]
        return pd.DataFrame(df), np.array(y)

    @staticmethod
    def fit(component, X, y, success):
        try:
            component().fit(X, y).transform(X)
            if not success:
                raise AssertionError(f'Expected failure {component}')
        except Exception as ex:
            if success:
                print(ex)
                raise AssertionError(f'Expected success {component}')

    def test_synthetic(self):
        for name, component in CapabilitiesTest.components.items():
            print(name)
            for X_req in itertools.product([False, True], repeat=3):
                if sum(X_req[:-1]) == 0:
                    continue

                for y_req in itertools.product([False, True], repeat=2):
                    X, y = self.synthetic_df(*X_req, *y_req)
                    prop = component.get_properties()
                    self.fit(component, X.copy(deep=True), np.copy(y),
                             (prop[HANDLES_NUMERIC] or not X_req[0]) and
                             (prop[HANDLES_NOMINAL] or not X_req[1]) and
                             (prop[HANDLES_MISSING] or not X_req[2]) and
                             (prop[HANDLES_NOMINAL_CLASS] or not y_req[1]))

    def test_component_selection(self):
        for choice in [ClassifierChoice(), DataPreprocessorChoice(), FeaturePreprocessorChoice()]:
            for X_req in itertools.product([False, True], repeat=3):
                if sum(X_req[:-1]) == 0:
                    continue

                for y_req in itertools.product([False, True], repeat=1):
                    # We do not have a meta-feature to check for nominal/numeric class
                    X, y = self.synthetic_df(*X_req, y_req[0], False)
                    mf = MetaFeatureFactory._calculate(X.to_numpy(), y)
                    assert mf is not None
                    
                    print(*X_req, y_req[0], False)

                    tested = []
                    for name, component in choice.get_available_components(mf=mf).items():
                        print(name)
                        self.fit(component, X.copy(deep=True), np.copy(y), True)
                        tested.append(name)
                    for name, component in choice.get_available_components(exclude=tested).items():
                        print(name)
                        self.fit(component, X.copy(deep=True), np.copy(y), False)

    def test_1510(self):
        X, y = datasets.fetch_openml(data_id=1510, return_X_y=True, as_frame=True)

        for name, component in CapabilitiesTest.components.items():
            print(name)
            prop = component.get_properties()
            self.fit(component, np.copy(X), np.copy(y), prop[HANDLES_NUMERIC])

    def test_40975(self):
        X, y = datasets.fetch_openml(data_id=40975, return_X_y=True, as_frame=True)

        for name, component in CapabilitiesTest.components.items():
            print(name)
            prop = component.get_properties()
            self.fit(component, np.copy(X), np.copy(y), prop[HANDLES_NOMINAL] and prop[HANDLES_MULTICLASS])

    def test_1480(self):
        X, y = datasets.fetch_openml(data_id=1480, return_X_y=True, as_frame=True)

        for name, component in CapabilitiesTest.components.items():
            print(name)
            prop = component.get_properties()
            self.fit(component, np.copy(X), np.copy(y), prop[HANDLES_NUMERIC] and prop[HANDLES_NOMINAL])

    def test_61(self):
        X, y = datasets.fetch_openml(data_id=61, return_X_y=True, as_frame=True)
        y_enc = LabelEncoder().fit_transform(y)

        for name, component in CapabilitiesTest.components.items():
            print(name)
            prop = component.get_properties()
            self.fit(component, np.copy(X), np.copy(y),
                     prop[HANDLES_NUMERIC] and prop[HANDLES_MULTICLASS] and prop[HANDLES_NOMINAL_CLASS])
            self.fit(component, np.copy(X), np.copy(y_enc), prop[HANDLES_NUMERIC] and prop[HANDLES_MULTICLASS])

    def test_15(self):
        X, y = datasets.fetch_openml(data_id=15, return_X_y=True, as_frame=True)
        y_enc = LabelEncoder().fit_transform(y)

        for name, component in CapabilitiesTest.components.items():
            print(name)
            prop = component.get_properties()
            self.fit(component, np.copy(X), np.copy(y),
                     prop[HANDLES_NUMERIC] and prop[HANDLES_MISSING] and prop[HANDLES_NOMINAL_CLASS])
            self.fit(component, np.copy(X), np.copy(y_enc), prop[HANDLES_NUMERIC] and prop[HANDLES_MISSING])

    def test_23381(self):
        X, y = datasets.fetch_openml(data_id=23381, return_X_y=True, as_frame=True)

        for name, component in CapabilitiesTest.components.items():
            print(name)
            prop = component.get_properties()
            self.fit(component, np.copy(X), np.copy(y),
                     prop[HANDLES_NUMERIC] and prop[HANDLES_NOMINAL] and prop[HANDLES_MISSING])
