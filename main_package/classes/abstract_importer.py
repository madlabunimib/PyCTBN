from abc import ABC, abstractmethod


class AbstractImporter(ABC):
    """
    Interface that exposes all the necessary methods to import the trajectories and the net structure.

    :file_path: the file path

    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        super().__init__()

    @abstractmethod
    def import_trajectories(self, raw_data):
        pass

    @abstractmethod
    def import_structure(self, raw_data):
        pass

