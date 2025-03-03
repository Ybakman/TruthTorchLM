from .normalizer import Normalizer
import sklearn.isotonic as isotonic


class IsotonicRegression(
    Normalizer
):  # see: https://scikit-learn.org/stable/modules/isotonic.html

    def __init__(self, y_min: float = 0.0, y_max: float = 1.0):
        self.iso_reg = isotonic.IsotonicRegression(
            y_min=y_min, y_max=y_max, out_of_bounds="clip"
        )
        self.iso_reg.fit([0.5], [1])  # dummy fit

    def fit(self, generation_performance_scores: list, truth_values: list):
        self.iso_reg.fit(truth_values, generation_performance_scores)
        print(
            f"Calibrated with the following parameters: {self.iso_reg.get_params()}")

    def __call__(self, truth_value):
        return self.iso_reg.predict([truth_value])[0]
