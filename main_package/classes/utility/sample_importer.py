import json
import typing

import pandas as pd
import numpy as np
import sys
sys.path.append('../')

import utility.abstract_importer as ai



class SampleImporter(ai.AbstractImporter):
    #TODO: Scrivere documentazione
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

    def __init__(self, trajectory_list: typing.Union[pd.DataFrame, np.ndarray] = None,
                 variables: pd.DataFrame = None, prior_net_structure: pd.DataFrame = None):
        super(SampleImporter, self).__init__(trajectory_list =trajectory_list,
                                        variables= variables,
                                        prior_net_structure=prior_net_structure)

    def import_data(self, header_column = None):

        if header_column is None:
            self._sorter = header_column
        else:    
            self._sorter = self.build_sorter(self._df_samples_list[0])

        samples_list= self._df_samples_list

        if isinstance(samples_list, np.ndarray):
            samples_list = samples_list.tolist()

        self.compute_row_delta_in_all_samples_frames(samples_list)

    def build_sorter(self, sample_frame: pd.DataFrame) -> typing.List:
        """Implements the abstract method build_sorter of the :class:`AbstractImporter` for this dataset.
        """
        columns_header = list(sample_frame.columns.values)
        del columns_header[0]
        return columns_header


    def dataset_id(self) -> object:
        pass