
import json_importer as imp
import trajectory as tr
import structure as st


class SamplePath:
    """
    Contiene l'aggregazione di una o più traiettorie e la struttura della rete.
    Ha il compito dato di costruire tutte gli oggetti Trajectory e l'oggetto Structure
    a partire dai dataframe contenuti in self.importer


    :importer: l'oggetto Importer che ha il compito di caricare i dataset
    :trajectories: lista di oggetti Trajectories
    :structure: oggetto Structure
    """

    def __init__(self, files_path, samples_label, structure_label, variables_label, time_key, variables_key):
        self.importer = imp.JsonImporter(files_path, samples_label, structure_label,
                                         variables_label, time_key, variables_key)
        self._trajectories = None
        self._structure = None
        self.total_variables_count = None

    def build_trajectories(self):
        self.importer.import_data()
        self._trajectories = \
            tr.Trajectory(self.importer.build_list_of_samples_array(self.importer.concatenated_samples),
                          len(self.importer.sorter) + 1)
        #self.trajectories.append(trajectory)
        self.importer.clear_concatenated_frame()

    def build_structure(self):
        self.total_variables_count = len(self.importer.sorter)
        self._structure = st.Structure(self.importer.structure, self.importer.variables,
                                       self.total_variables_count)

    @property
    def trajectories(self):
        return self._trajectories

    @property
    def structure(self):
        return self._structure

    def total_variables_count(self):
        return self.total_variables_count

    """def build_possible_values_variables_structure(self):
        possible_val_list = []
        print(self.importer.variables)
        for cardinality in self.importer.variables['Value']:
            possible_val_list.append(list(range(0, cardinality)))
        self.possible_variables_values = possible_val_list"""




