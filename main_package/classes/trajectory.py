
import numpy as np


class Trajectory:
    """ 
    Abstracts the infos about a complete set of trajectories, represented as a numpy array of doubles and a numpy matrix
    of ints.

    :list_of_columns: the list containing the times array and values matrix
    :original_cols_numb: total number of cols in the data
    :actual_trajectory: the trajectory containing also the duplicated and shifted values
    :times: the array containing the time deltas

    """

    def __init__(self, list_of_columns, original_cols_number):
        if type(list_of_columns[0][0]) != np.float64:
            raise TypeError('The first array in the list has to be Times')
        self.original_cols_number = original_cols_number
        self._actual_trajectory = np.array(list_of_columns[1:], dtype=np.int).T
        self._times = np.array(list_of_columns[0], dtype=np.float)

    @property
    def trajectory(self) -> np.ndarray:
        """
        Parameters:
            void
        Returns:
            a numpy matrix containing ONLY the original columns values, not the shifted ones
        """
        return self._actual_trajectory[:, :self.original_cols_number]

    @property
    def complete_trajectory(self) -> np.ndarray:
        """
                Parameters:
                    void
                Returns:
                    a numpy matrix containing all the values
                """
        return self._actual_trajectory

    @property
    def times(self):
        return self._times

    def size(self):
        return self._actual_trajectory.shape[0]

    def __repr__(self):
        return "Complete Trajectory Rows: " + str(self.size()) + "\n" + self.complete_trajectory.__repr__() + \
               "\nTimes Rows:" + str(self.times.size) + "\n" + self.times.__repr__()


