
import abstract_importer as imp
import structure as st
import trajectory as tr


class SamplePath:
    """
    Aggregates all the informations about the trajectories, the real structure of the sampled net and variables
    cardinalites.
    Has the task of creating the objects that will contain the mentioned data.


    :trajectories: the Trajectory object that will contain all the concatenated trajectories
    :structure: the Structure Object that will contain all the structurral infos about the net
    :total_variables_count: the number of variables in the net

    """
    def __init__(self, importer: imp.AbstractImporter):
        """
        :importer: the Importer objects that will import ad process data
        """
        self.importer = importer
        self._trajectories = None
        self._structure = None
        self.total_variables_count = None

    def build_trajectories(self):
        """
        Builds the Trajectory object that will contain all the trajectories.
        Clears all the unused dataframes in Importer Object

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
        if self.importer.sorter != self.importer.variables.iloc[:, 0].to_list():
            raise RuntimeError("The Dataset columns order have to match the order of labels in the variables Frame!")
        self.total_variables_count = len(self.importer.sorter)
        #labels = self.importer.variables[self.importer.variables_key].to_list()
        #print("SAMPLE PATH LABELS",labels)
        #print(self.importer.variables)
        labels = self.importer.variables.iloc[:, 0].to_list()
        indxs = self.importer.variables.index.to_numpy()
        vals = self.importer.variables.iloc[:, 1].to_numpy()
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





