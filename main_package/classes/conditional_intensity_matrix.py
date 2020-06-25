import numpy as np


class ConditionalIntensityMatrix:

    def __init__(self, dimension):
        self.state_residence_times = np.zeros(shape=(1, dimension))
        self.state_transition_matrix = np.zeros(shape=(dimension, dimension))
        self.cim = np.zeros(shape=(dimension, dimension))