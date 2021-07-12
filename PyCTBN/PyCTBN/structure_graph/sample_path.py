
# License: MIT License



import numpy as np
import pandas as pd

from .structure import Structure
from .trajectory import Trajectory
from ..utility.abstract_importer import AbstractImporter


MESSAGE_HAS_TO_CONTAIN_EXCEPTION = 'The importer object has to contain the all processed data!'


class SamplePath(object):
    """Aggregates all the informations about the trajectories, the real structure of the sampled net and variables
    cardinalites. Has the task of creating the objects ``Trajectory`` and ``Structure`` that will
    contain the mentioned data.

    :param importer: the Importer object which contains the imported and processed data
    :type importer: AbstractImporter
    :_trajectories: the ``Trajectory`` object that will contain all the concatenated trajectories
    :_structure: the ``Structure`` Object that will contain all the structural infos about the net
    :_total_variables_count: the number of variables in the net
    """
    def __init__(self, importer: AbstractImporter):
        """Constructor Method
        """
        self._importer = importer
        if self._importer._df_variables is None or self._importer._concatenated_samples is None:
            raise RuntimeError()
        if self._importer._df_variables.empty:
            raise RuntimeError(MESSAGE_HAS_TO_CONTAIN_EXCEPTION)
        if isinstance(self._importer._concatenated_samples, pd.DataFrame) and\
            self._importer._concatenated_samples.empty:
                raise RuntimeError(MESSAGE_HAS_TO_CONTAIN_EXCEPTION)
        if isinstance(self._importer._concatenated_samples, np.ndarray) and\
            self._importer._concatenated_samples.size == 0:
                raise RuntimeError(MESSAGE_HAS_TO_CONTAIN_EXCEPTION)
        self._trajectories = None
        self._structure = None
        self._total_variables_count = None

    def build_trajectories(self) -> None:
        """Builds the Trajectory object that will contain all the trajectories.
        Clears all the unused dataframes in ``_importer`` Object
        """
        self._trajectories = \
            Trajectory(self._importer.build_list_of_samples_array(self._importer.concatenated_samples),
                len(self._importer.sorter) + 1)
        self._importer.clear_concatenated_frame()

    def build_structure(self) -> None:
        """
        Builds the ``Structure`` object that aggregates all the infos about the net.
        """
        if self._importer.sorter != self._importer.variables.iloc[:, 0].to_list():
            raise RuntimeError("The Dataset columns order have to match the order of labels in the variables Frame!")

        self._total_variables_count = len(self._importer.sorter)
        labels = self._importer.variables.iloc[:, 0].to_list()
        indxs = self._importer.variables.index.to_numpy()
        vals = self._importer.variables.iloc[:, 1].to_numpy()
        if self._importer.structure is None or self._importer.structure.empty:
            edges = []
        else:
            edges = list(self._importer.structure.to_records(index=False))
        self._structure = Structure(labels, indxs, vals, edges,
                                       self._total_variables_count)

    def clear_memory(self):
        self._importer._raw_data = [] 

    @property
    def trajectories(self) -> Trajectory:
        return self._trajectories

    @property
    def structure(self) -> Structure:
        return self._structure

    @property
    def total_variables_count(self) -> int:
        return self._total_variables_count

    @property
    def has_prior_net_structure(self) -> bool:
        return bool(self._structure.edges)






