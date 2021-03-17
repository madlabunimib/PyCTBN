import numpy as np


class ConditionalIntensityMatrix(object):
    """Abstracts the Conditional Intesity matrix of a node as aggregation of the state residence times vector
    and state transition matrix and the actual CIM matrix.

    :param state_residence_times: state residence times vector
    :type state_residence_times: numpy.array
    :param state_transition_matrix: the transitions count matrix
    :type state_transition_matrix: numpy.ndArray
    :_cim: the actual cim of the node
    """
    def __init__(self, state_residence_times: np.array = None, state_transition_matrix: np.array = None, 
        cim: np.array = None):
        """Constructor Method
        """
        self._state_residence_times = state_residence_times
        self._state_transition_matrix = state_transition_matrix
        if cim is not None:
            self._cim = cim
        else:
            self._cim = self.state_transition_matrix.astype(np.float64)

    def compute_cim_coefficients(self) -> None:
        """Compute the coefficients of the matrix _cim by using the following equality q_xx' = M[x, x'] / T[x].
        The class member ``_cim`` will contain the computed cim
        """
        np.fill_diagonal(self._cim, self._cim.diagonal() * -1)
        self._cim = ((self._cim.T + 1) / (self._state_residence_times + 1)).T

    @property
    def state_residence_times(self) -> np.ndarray:
        return self._state_residence_times

    @property
    def state_transition_matrix(self) -> np.ndarray:
        return self._state_transition_matrix

    @property
    def cim(self) -> np.ndarray:
        return self._cim

    def __repr__(self):
        return 'CIM:\n' + str(self.cim)

