
import json
import typing

import pandas as pd

import abstract_importer as ai


class JsonImporter(ai.AbstractImporter):
    """
    Implements the Interface AbstractImporter and adds all the necessary methods to process and prepare the data in json ext.
    with the following structure:
    [] 0
        |_ dyn.cims
        |_ dyn.str
        |_ samples
        |_ variabels
    :file_path: the path of the file that contains tha data to be imported
    :samples_label: the reference key for the samples in the trajectories
    :structure_label: the reference key for the structure of the network data
    :variables_label: the reference key for the cardinalites of the nodes data
    :time_key: the key used to identify the timestamps in each trajectory
    :variables_key: the key used to identify the names of the variables in the net
    :df_samples_list: a Dataframe list in which every df contains a trajectory
    """

    def __init__(self, file_path: str, samples_label: str, structure_label: str, variables_label: str, time_key: str,
                 variables_key: str):
        self.samples_label = samples_label
        self.structure_label = structure_label
        self.variables_label = variables_label
        self.time_key = time_key
        self.variables_key = variables_key
        self.df_samples_list = []
        super(JsonImporter, self).__init__(file_path)

    def import_data(self):
        """
        Imports and prepares all data present needed for susequent computation.
        Parameters:
            void
        Returns:
            void
        """
        raw_data = self.read_json_file()
        self.import_trajectories(raw_data)
        self.compute_row_delta_in_all_samples_frames(self.df_samples_list)
        self.clear_data_frame_list()
        self.import_structure(raw_data)
        self.import_variables(raw_data, self._sorter)

    def import_trajectories(self, raw_data: typing.List):
        """
        Imports the trajectories in the list of dicts raw_data.
        Parameters:
            :raw_data: List of Dicts
        Returns:
            void
        """
        self.normalize_trajectories(raw_data, 0, self.samples_label)

    def import_structure(self, raw_data: typing.List):
        """
        Imports in a dataframe the data in the list raw_data at the key structure_label

        Parameters:
            raw_data: the data
        Returns:
            void
        """
        self._df_structure = self.one_level_normalizing(raw_data, 0, self.structure_label)

    def import_variables(self, raw_data: typing.List, sorter: typing.List):
        """
        Imports the data in raw_data at the key variables_label.
        Sorts the row of the dataframe df_variables using the list sorter.

        Parameters:
            raw_data: the data
            sorter: the list used to sort the dataframe self.df_variables
        Returns:
            void
        """
        self._df_variables = self.one_level_normalizing(raw_data, 0, self.variables_label)
        #TODO Usando come Pre-requisito l'ordinamento del frame _df_variables uguale a quello presente in
        #TODO self _sorter questo codice risulta inutile
        self._df_variables[self.variables_key] = self._df_variables[self.variables_key].astype("category")
        self._df_variables[self.variables_key] = self._df_variables[self.variables_key].cat.set_categories(sorter)
        self._df_variables = self._df_variables.sort_values([self.variables_key])
        self._df_variables.reset_index(inplace=True)
        self._df_variables.drop('index', axis=1, inplace=True)
        print("Var Frame", self._df_variables)

    def read_json_file(self) -> typing.List:
        """
        Reads the first json file in the path self.filePath

        Parameters:
              void
        Returns:
              data: the contents of the json file

        """
        with open(self.file_path) as f:
            data = json.load(f)
            return data

    def one_level_normalizing(self, raw_data: typing.List, indx: int, key: str) -> pd.DataFrame:
        """
        Extracts the one-level nested data in the list raw_data at the index indx at the key key

        Parameters:
            raw_data: List of Dicts
            indx: The index of the array from which the data have to be extracted
            key: the key for the Dicts from which exctract data
        Returns:
            a normalized dataframe

        """
        return pd.DataFrame(raw_data[indx][key])

    def normalize_trajectories(self, raw_data: typing.List, indx: int, trajectories_key: str):
        """
        Extracts the traj in raw_data at the index index at the key trajectories key.
        Adds the extracted traj in the dataframe list self._df_samples_list.
        Initializes the list self.sorter.

        Parameters:
            raw_data: the data
            indx: the index of the array from which extract data
            trajectories_key: the key of the trajectories objects
        Returns:
            void
        """
        dataframe = pd.DataFrame
        smps = raw_data[indx][trajectories_key]
        self.df_samples_list = [dataframe(sample) for sample in smps]
        columns_header = list(self.df_samples_list[0].columns.values)
        #print("COLUMNs HEADER", columns_header)
        columns_header.remove(self.time_key)
        self._sorter = columns_header

    def clear_data_frame_list(self):
        """
        Removes all values present in the dataframes in the list df_samples_list
        """
        for indx in range(len(self.df_samples_list)):
            self.df_samples_list[indx] = self.df_samples_list[indx].iloc[0:0]

    def import_sampled_cims(self, raw_data: typing.List, indx: int, cims_key: str) -> typing.Dict:
        cims_for_all_vars = {}
        for var in raw_data[indx][cims_key]:
            sampled_cims_list = []
            cims_for_all_vars[var] = sampled_cims_list
            for p_comb in raw_data[indx][cims_key][var]:
                cims_for_all_vars[var].append(pd.DataFrame(raw_data[indx][cims_key][var][p_comb]).to_numpy())
        return cims_for_all_vars





