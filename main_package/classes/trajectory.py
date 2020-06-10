import pandas as pd
import numpy as np


class Trajectory():
    """ 
    Rappresenta una traiettoria come un data_frame contenente coppie (T_k,S_i) => (numpy.float64, string)
    Offre i metodi utili alla computazione sulla struttura stessa.
    :actual_trajectory: il data_frame contenente la successione di coppie (T_k,S_i)

    """

    def __init__(self, data_frame):
        self.actual_trajectory = data_frame
        
    def get_trajectory(self):
        return self.actual_trajectory

    def get_trajectory_as_matrix(self):
        """
        Converte il data_frame actual_trajectory in formato numpy.array
        Parameters:
            void
        Returns:
            numpy.array
        """
        return self.actual_trajectory[['Time','State']].to_numpy()

    def get_states(self):
        """
        Identifica gli stati visitati nella traiettoria.
        Parameters:
            void
        Returns:
            una lista contenente gli stati visitati nella traiettoria
        """
        return self.actual_trajectory['State'].unique()
