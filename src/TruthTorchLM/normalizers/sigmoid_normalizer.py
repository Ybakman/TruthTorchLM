from .normalizer import Normalizer
from TruthTorchLM.utils.calibration_utils import f1_picker
from TruthTorchLM.utils.common_utils import sigmoid_normalization
import numpy as np

class SigmoidNormalizer(Normalizer):

    def __init__(self, threshold:float = 0.0, std:float = 1.0, threshold_picker = f1_picker):
        self.threshold_picker = threshold_picker
        self.threshold = threshold
        self.std = std

    def fit(self, generation_performance_scores:list, truth_values:list):
        self.threshold = self.threshold_picker(generation_performance_scores, truth_values)
        self.std = np.std(truth_values)
        print('Calibrated with the following parameters: threshold =', self.threshold, 'std =', self.std)

    def __call__(self, truth_value):
        return sigmoid_normalization(truth_value, self.threshold, self.std)
    

    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def set_std(self, std):
        self.std = std

    def get_threshold(self):
        return self.threshold
    
    def get_std(self):
        return self.std
    
        