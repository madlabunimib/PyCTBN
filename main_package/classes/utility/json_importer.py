import json
import typing

import pandas as pd
import sys
sys.path.append('../')

import utility.abstract_importer as ai


class JsonImporter(ai.AbstractImporter):
    """Implements the abstracts methods of AbstractImporter and adds all the necessary methods to process and prepare
    the data in json extension.

    :param file_path: the path of the file that contains tha data to be imported
    :type file_path: string
    :param samples_label: the reference key for the samples in the trajectories
    :type samples_label: string
    :param structure_label: the reference key for the structure of the network data
    :type structure_label: string
    :param variables_label: the reference key for the cardinalites of the nodes data
    :type variables_label: string
    :param time_key: the key used to identify the timestamps in each trajectory
    :type time_key: string
    :param variables_key: the key used to identify the names of the variables in the net
    :type variables_key: string
    :_array_indx: the index of the outer JsonArray to extract the data from
    :type _array_indx: int
    :_df_samples_list: a Dataframe list in which every dataframe contains a trajectory
    :_raw_data: The raw contents of the json file to import
    :type _raw_data: List
    """

    def __init__(self, file_path: str, samples_label: str, structure_label: str, variables_label: str, time_key: str,
                 variables_key: str):
        """Constructor method

        .. note::
            This constructor calls also the method ``read_json_file()``, so after the construction of the object
            the class member ``_raw_data`` will contain the raw imported json data.

        """
        self._samples_label = samples_label
        self._structure_label = structure_label
        self._variables_label = variables_label
        self._time_key = time_key
        self._variables_key = variables_key
        self._df_samples_list = None
        self._array_indx = None
        super(JsonImporter, self).__init__(file_path)
        self._raw_data = self.read_json_file()

    def import_data(self, indx: int) -> None:
        """Implements the abstract method of :class:`AbstractImporter`.

        :param indx: the index of the outer JsonArray to extract the data from
        :type indx: int
        """
        self._array_indx = indx
        self._df_samples_list = self.import_trajectories(self._raw_data)
        self._sorter = self.build_sorter(self._df_samples_list[0])
        self.compute_row_delta_in_all_samples_frames(self._df_samples_list)
        self.clear_data_frame_list()
        self._df_structure = self.import_structure(self._raw_data)
        self._df_variables = self.import_variables(self._raw_data)

    def import_trajectories(self, raw_data: typing.List) -> typing.List:
        """Imports the trajectories from the list of dicts ``raw_data``.

        :param raw_data: List of Dicts
        :type raw_data: List
        :return: List of dataframes containing all the trajectories
        :rtype: List
        """
        return self.normalize_trajectories(raw_data, self._array_indx, self._samples_label)

    def import_structure(self, raw_data: typing.List) -> pd.DataFrame:
        """Imports in a dataframe the data in the list raw_data at the key ``_structure_label``

        :param raw_data: List of Dicts
        :type raw_data: List
        :return: Dataframe containg the starting node a ending node of every arc of the network
        :rtype: pandas.Dataframe
        """
        return self.one_level_normalizing(raw_data, self._array_indx, self._structure_label)

    def import_variables(self, raw_data: typing.List) -> pd.DataFrame:
        """Imports the data in ``raw_data`` at the key ``_variables_label``.

        :param raw_data: List of Dicts
        :type raw_data: List
        :return: Datframe containg the variables simbolic labels and their cardinalities
        :rtype: pandas.Dataframe
        """
        return self.one_level_normalizing(raw_data, self._array_indx, self._variables_label)

    def read_json_file(self) -> typing.List:
        """Reads the JSON file in the path self.filePath.

        :return: The contents of the json file
        :rtype: List
        """
        with open(self._file_path) as f:
            data = json.load(f)
            return data

    def one_level_normalizing(self, raw_data: typing.List, indx: int, key: str) -> pd.DataFrame:
        """Extracts the one-level nested data in the list ``raw_data`` at the index ``indx`` at the key ``key``.

        :param raw_data: List of Dicts
        :type raw_data: List
        :param indx: The index of the array from which the data have to be extracted
        :type indx: int
        :param key: the key for the Dicts from which exctract data
        :type key: string
        :return: A normalized dataframe
        :rtype: pandas.Datframe
        """
        return pd.DataFrame(raw_data[indx][key])

    def normalize_trajectories(self, raw_data: typing.List, indx: int, trajectories_key: str) -> typing.List:
        """
        Extracts the trajectories in ``raw_data`` at the index ``index`` at the key ``trajectories key``.

        :param raw_data: List of Dicts
        :type raw_data: List
        :param indx: The index of the array from which the data have to be extracted
        :type indx: int
        :param trajectories_key: the key of the trajectories objects
        :type trajectories_key: string
        :return: A list of daframes containg the trajectories
        :rtype: List
        """
        dataframe = pd.DataFrame
        smps = raw_data[indx][trajectories_key]
        df_samples_list = [dataframe(sample) for sample in smps]
        return df_samples_list

    def build_sorter(self, sample_frame: pd.DataFrame) -> typing.List:
        """Implements the abstract method build_sorter of the :class:`AbstractImporter` for this dataset.
        """
        columns_header = list(sample_frame.columns.values)
        columns_header.remove(self._time_key)
        return columns_header

    def clear_data_frame_list(self) -> None:
        """Removes all values present in the dataframes in the list ``_df_samples_list``.
        """
        for indx in range(len(self._df_samples_list)):
            self._df_samples_list[indx] = self._df_samples_list[indx].iloc[0:0]

    def dataset_id(self) -> object:
        return self._array_indx

    def import_sampled_cims(self, raw_data: typing.List, indx: int, cims_key: str) -> typing.Dict:
        """Imports the synthetic CIMS in the dataset in a dictionary, using variables labels
        as keys for the set of CIMS of a particular node.

        :param raw_data: List of Dicts
        :type raw_data: List
        :param indx: The index of the array from which the data have to be extracted
        :type indx: int
        :param cims_key: the key where the json object cims are placed
        :type cims_key: string
        :return: a dictionary containing the sampled CIMS for all the variables in the net
        :rtype: Dictionary
        """
        cims_for_all_vars = {}
        for var in raw_data[indx][cims_key]:
            sampled_cims_list = []
            cims_for_all_vars[var] = sampled_cims_list
            for p_comb in raw_data[indx][cims_key][var]:
                cims_for_all_vars[var].append(pd.DataFrame(raw_data[indx][cims_key][var][p_comb]).to_numpy())
        return cims_for_all_vars



