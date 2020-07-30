
import numpy as np


class Trajectory:
    """ 
    Rappresenta una traiettoria come un numpy_array contenente n-ple (indx, T_k,S_i,.....,Sj)
    Offre i metodi utili alla computazione sulla struttura stessa.

    Una Trajectory viene costruita a partire da una lista di numpyarray dove ogni elemento rappresenta una colonna
    della traj

    :actual_trajectory: il numpy_array contenente la successione di n-ple (indx, T_k,S_i,.....,Sj)

    """

    def __init__(self, list_of_columns, original_cols_number):
        if type(list_of_columns[0][0]) != np.float64:
            raise TypeError('The first array in the list has to be Times')
        self.original_cols_number = original_cols_number
        self._actual_trajectory = np.array(list_of_columns[1:], dtype=np.int).T
        self._times = np.array(list_of_columns[0], dtype=np.float)

    @property
    def trajectory(self):
        return self._actual_trajectory[:, :self.original_cols_number]

    @property
    def complete_trajectory(self):
        return self._actual_trajectory

    @property
    def times(self):
        return self._times

    def size(self):
        return self._actual_trajectory.shape[0]

    def __repr__(self):
        return "Complete Trajectory Rows: " + str(self.size()) + "\n" + self.complete_trajectory.__repr__() + \
               "\nTimes Rows:" + str(self.times.size) + "\n" + self.times.__repr__()


