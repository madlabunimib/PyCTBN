import pandas as pd
import numpy as np


class Trajectory():
    """ 
    Rappresenta una traiettoria come un numpy_array contenente n-ple (indx, T_k,S_i,.....,Sj)
    Offre i metodi utili alla computazione sulla struttura stessa.

    Una Trajectory viene costruita a partire da una lista di numpyarray dove ogni elemento rappresenta una colonna
    della traj

    :actual_trajectory: il numpy_array contenente la successione di n-ple (indx, T_k,S_i,.....,Sj)

    """

    def __init__(self, list_of_columns):
        self.actual_trajectory = np.array(list_of_columns, dtype=object).T
        
    def get_trajectory(self):
        return self.actual_trajectory

    def size(self):
        return self.actual_trajectory.shape[0]

    def merge_columns(self, list_of_cols):
        return np.vstack(list_of_cols).T

