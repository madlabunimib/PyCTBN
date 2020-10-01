from abc import ABC, abstractmethod

import abstract_importer as ai


class AbstractSamplePath(ABC):

    def __init__(self, importer: ai.AbstractImporter):
        self.importer = importer
        self._trajectories = None
        self._structure = None
        super().__init__()

    @abstractmethod
    def build_trajectories(self):
        """
        Builds the Trajectory object that will contain all the trajectories.
        Assigns the Trajectoriy object to the instance attribute _trajectories
        Clears all the unused dataframes in Importer Object

        Parameters:
            void
        Returns:
            void
        """
        pass

    @abstractmethod
    def build_structure(self):
        """
        Builds the Structure object that aggregates all the infos about the net.
        Assigns the Structure object to the instance attribuite _structure
        Parameters:
            void
        Returns:
            void
        """
        pass
