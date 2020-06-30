from abc import ABC, abstractmethod


class AbstractImporter(ABC):
    """
    Interfaccia che espone i metodi necessari all'importing delle trajectories e della struttura della CTBN

    :files_path: il path in cui sono presenti i/il file da importare

    """

    def __init__(self, files_path):
        self.files_path = files_path
        super().__init__()

    @abstractmethod
    def import_trajectories(self, raw_data):
        """
        Costruisce le traj partendo dal dataset raw_data
        Parameters:
            raw_data: il dataset da cui estrarre le traj
        Returns:
            void
        """
        pass

    @abstractmethod
    def import_structure(self, raw_data):
        """
        Costruisce la struttura della rete partendo dal dataset raw_data
        Parameters:
            raw_data: il dataset da cui estrarre la struttura
        Returns:
            void
        """
        pass
