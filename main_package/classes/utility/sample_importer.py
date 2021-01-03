import json
import typing

import pandas as pd
import numpy as np
import sys
sys.path.append('../')

import utility.abstract_importer as ai



class SampleImporter(ai.AbstractImporter):
    """Implements the abstracts methods of AbstractImporter and adds all the necessary methods to process and prepare
    the data loaded directly by using DataFrame

    :param trajectory_list: the data that describes the trajectories
    :type trajectory_list: typing.Union[pd.DataFrame, np.ndarray, typing.List]
    :param variables: the data that describes the variables with name and cardinality
    :type variables: typing.Union[pd.DataFrame, np.ndarray, typing.List]
    :param prior_net_structure: the data of the real structure, if it exists
    :type prior_net_structure: typing.Union[pd.DataFrame, np.ndarray, typing.List]

    :_df_samples_list: a Dataframe list in which every dataframe contains a trajectory
    :_raw_data: The raw contents of the json file to import
    :type _raw_data: List
    """

    def __init__(self, 
                trajectory_list: typing.Union[pd.DataFrame, np.ndarray, typing.List] = None,
                variables: typing.Union[pd.DataFrame, np.ndarray, typing.List] = None,
                prior_net_structure: typing.Union[pd.DataFrame, np.ndarray,typing.List] = None):

        'If the data are not DataFrame, it will be converted'
        if isinstance(variables,list) or isinstance(variables,np.ndarray):
            variables = pd.DataFrame(variables)
        if isinstance(variables,list) or isinstance(variables,np.ndarray):
            prior_net_structure=pd.DataFrame(prior_net_structure)

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
        """Implements the abstract method build_sorter of the :class:`AbstractImporter` in order to get the ordered variables list.
        """
        columns_header = list(sample_frame.columns.values)
        del columns_header[0]
        return columns_header


    def dataset_id(self) -> object:
        pass