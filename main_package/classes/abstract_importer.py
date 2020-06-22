from abc import ABC, abstractmethod


class AbstractImporter(ABC):

    def __init__(self, files_path):
        self.files_path = files_path
        super().__init__()

    @abstractmethod
    def import_trajectories(self, raw_data):
        pass

    @abstractmethod
    def import_structure(self, raw_data):
        pass
