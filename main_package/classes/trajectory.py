
import numpy as np


class Trajectory:
    """ 
    Rappresenta una traiettoria come un numpy_array contenente n-ple (indx, T_k,S_i,.....,Sj)
    Offre i metodi utili alla computazione sulla struttura stessa.

    Una Trajectory viene costruita a partire da una lista di numpyarray dove ogni elemento rappresenta una colonna
    della traj

    :actual_trajectory: il numpy_array contenente la successione di n-ple (indx, T_k,S_i,.....,Sj)

    """

    def __init__(self, list_of_columns):
        print(list_of_columns)
        self._actual_trajectory = np.array(list_of_columns[1:], dtype=np.int).T
        self._times = np.array(list_of_columns[0], dtype=np.float)
        print(self._times)

    @property
    def trajectory(self):
        return self._actual_trajectory[:, :4]

    @property
    def complete_trajectory(self):
        return self._actual_trajectory

    @property
    def times(self):
        return self._times

    def size(self):
        return self.actual_trajectory.shape[0]


