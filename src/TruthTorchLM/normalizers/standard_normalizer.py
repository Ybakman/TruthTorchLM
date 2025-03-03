from .normalizer import Normalizer
from sklearn import preprocessing


class StandardNormalizer(Normalizer):

    def __init__(self):
        self.standard_scaler = preprocessing.StandardScaler()
        self.standard_scaler.fit([[0], [1]])  # dummy fit

    def fit(self, generation_performance_scores: list, truth_values: list):
        self.standard_scaler.fit(truth_values.reshape(-1, 1))

    def __call__(self, truth_value):
        return self.standard_scaler.transform([[truth_value]])[0][0]
