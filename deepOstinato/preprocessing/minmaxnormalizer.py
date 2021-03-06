from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from deepOstinato.preprocessing.constants import MAX_VAL, MIN_VAL
from deepOstinato.preprocessing.saver import Saver

class MinMaxNormaliser(BaseEstimator, TransformerMixin):
    """Applies MinMaxNormalisation to an array.
    Takes minimum and maximum values from transformed audio"""
    def __init__(self, min_val = -2, max_val = 2, original_min = MIN_VAL, original_max = MAX_VAL):
        self.min_val = min_val
        self.max_val = max_val
        self.original_min = original_min
        self.original_max = original_max

    def fit(self):
        """Fit method for the normalizer"""
        return self

    def transform(self, array):
        """Transform method that takes an array and normalize """
        normalised_audio = (array - self.original_min) / (self.original_max - self.original_min)
        normalised_audio = (array - self.min_val) / (self.max_val - self.min_val)
        return normalised_audio


class MinMaxDenormaliser(BaseEstimator, TransformerMixin):
    """Retransform the array to its original scale."""

    def __init__(self, min_val =-2, max_val=2, original_min = MIN_VAL, original_max = MAX_VAL):
        self.min_val = min_val
        self.max_val = max_val
        self.original_min = original_min
        self.original_max = original_max

    def fit(self):
        """Fit method for the denormalizer"""
        return self

    def transform(self, normalised_array):
        """Transform method that takes a normalized array and transforms it to its original scale """
        denormalised_array = (normalised_array - self.min_val) / (self.max_val - self.min_val)
        denormalised_array = denormalised_array * (self.original_max - self.original_min) + self.original_min
        return denormalised_array
