
# License: MIT License


import typing

import numpy as np


class Trajectory(object):
    """ Abstracts the infos about a complete set of trajectories, represented as a numpy array of doubles
    (the time deltas) and a numpy matrix of ints (the changes of states).

    :param list_of_columns: the list containing the times array and values matrix
    :type list_of_columns: List
    :param original_cols_number: total number of cols in the data
    :type original_cols_number: int
    :_actual_trajectory: the trajectory containing also the duplicated/shifted values
    :_times: the array containing the time deltas
    """

    def __init__(self, list_of_columns: typing.List, original_cols_number: int):
        """Constructor Method
        """
        self._times = list_of_columns[0]
        self._actual_trajectory = list_of_columns[1]
        self._original_cols_number = original_cols_number

    @property
    def trajectory(self) -> np.ndarray:
        return self._actual_trajectory[:, :self._original_cols_number - 1]

    @property
    def complete_trajectory(self) -> np.ndarray:
        return self._actual_trajectory

    @property
    def times(self):
        return self._times

    def size(self):
        return self._actual_trajectory.shape[0]

    def __repr__(self):
        return "Complete Trajectory Rows: " + str(self.size()) + "\n" + self.complete_trajectory.__repr__() + \
               "\nTimes Rows:" + str(self.times.size) + "\n" + self.times.__repr__()


