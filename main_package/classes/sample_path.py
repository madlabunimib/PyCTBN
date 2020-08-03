
import json_importer as imp
import trajectory as tr
import structure as st


class SamplePath:
    """
    Aggregates all the informations about the trajectories, the real structure of the sampled net and variables
    cardinalites.
    Has the task of creating the objects that will contain the mentioned data.

    :files_path: the path that contains tha data to be imported
    :samples_label: the reference key for the samples in the trajectories
    :structure_label: the reference key for the structure of the network data
    :variables_label: the reference key for the cardinalites of the nodes data
    :time_key: the key used to identify the timestamps in each trajectory
    :variables_key: the key used to identify the names of the variables in the net

    :importer: the Importer objects that will import ad process data
    :trajectories: the Trajectory object that will contain all the concatenated trajectories
    :structure: the Structure Object that will contain all the structurral infos about the net
    :total_variables_count: the number of variables in the net

    """

    def __init__(self, files_path: str, samples_label: str, structure_label: str, variables_label: str, time_key: str,
                 variables_key: str):
        self.importer = imp.JsonImporter(files_path, samples_label, structure_label,
                                         variables_label, time_key, variables_key)
        self._trajectories = None
        self._structure = None
        self.total_variables_count = None

    def build_trajectories(self):
        """
        Builds the Trajectory object that will contain all the trajectories.
        Clears all the unsed dataframes in Importer Object

        Parameters:
            void
        Returns:
            void
        """
        self.importer.import_data()
        self._trajectories = \
            tr.Trajectory(self.importer.build_list_of_samples_array(self.importer.concatenated_samples),
                          len(self.importer.sorter) + 1)
        #self.trajectories.append(trajectory)
        self.importer.clear_concatenated_frame()

    def build_structure(self):
        """
        Builds the Structure object that aggregates all the infos about the net.
        Parameters:
            void
        Returns:
            void
        """
        self.total_variables_count = len(self.importer.sorter)
        labels = self.importer.variables[self.importer.variables_key].to_list()
        #print("SAMPLE PATH LABELS",labels)
        indxs = self.importer.variables.index.to_numpy()
        vals = self.importer.variables['Value'].to_numpy()
        edges = list(self.importer.structure.to_records(index=False))
        self._structure = st.Structure(labels, indxs, vals, edges,
                                       self.total_variables_count)

    @property
    def trajectories(self):
        return self._trajectories

    @property
    def structure(self):
        return self._structure

    def total_variables_count(self):
        return self.total_variables_count





