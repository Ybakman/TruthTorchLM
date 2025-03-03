from .normalizer import Normalizer
from sklearn import preprocessing


class MinMaxNormalizer(Normalizer):

    def __init__(self, float_range=(0, 1)):
        self.min_max_scaler = preprocessing.MinMaxScaler(
            feature_range=float_range)
        self.min_max_scaler.fit([[0], [1]])  # dummy fit

    def fit(self, generation_performance_scores: list, truth_values: list):
        self.min_max_scaler.fit(truth_values.reshape(-1, 1))

    def __call__(self, truth_value):
        return self.min_max_scaler.transform([[truth_value]])[0][0]
