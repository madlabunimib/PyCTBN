import sys
sys.path.append('../')

import json
import typing

import pandas as pd

import utility.abstract_importer as ai


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
    :df_structure: Dataframe containing the structure of the network (edges)
    :df_variables: Dataframe containing the nodes cardinalities
    :df_concatenated_samples: the concatenation and processing of all the trajectories present in the list df_samples list
    :sorter: the columns header(excluding the time column) of the Dataframe concatenated_samples
    """

    def __init__(self, file_path: str, samples_label: str, structure_label: str, variables_label: str, time_key: str,
                 variables_key: str, network_number:int=0):
        self.samples_label = samples_label
        self.structure_label = structure_label
        self.variables_label = variables_label
        self.time_key = time_key
        self.variables_key = variables_key
        self.df_samples_list = []
        self._df_structure = pd.DataFrame()
        self._df_variables = pd.DataFrame()
        self._concatenated_samples = None
        self.sorter = None
        self.network_number= network_number
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
        self.compute_row_delta_in_all_samples_frames(self.time_key)
        self.clear_data_frame_list()
        self.import_structure(raw_data)
        self.import_variables(raw_data, self.sorter)

    def import_trajectories(self, raw_data: typing.List):
        """
        Imports the trajectories in the list of dicts raw_data.
        Parameters:
            :raw_data: List of Dicts
        Returns:
            void
        """
        self.normalize_trajectories(raw_data, self.network_number, self.samples_label)

    def import_structure(self, raw_data: typing.List):
        """
        Imports in a dataframe the data in the list raw_data at the key structure_label

        Parameters:
            raw_data: the data
        Returns:
            void
        """
        self._df_structure = self.one_level_normalizing(raw_data, self.network_number, self.structure_label)


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
        self._df_variables = self.one_level_normalizing(raw_data, self.network_number, self.variables_label)
        #self.sorter = self._df_variables[self.variables_key].to_list()
        #self.sorter.sort()
        #print("Sorter:", self.sorter)
        self._df_variables[self.variables_key] = self._df_variables[self.variables_key].astype("category")
        self._df_variables[self.variables_key] = self._df_variables[self.variables_key].cat.set_categories(sorter)
        self._df_variables = self._df_variables.sort_values([self.variables_key])
        self._df_variables.reset_index(inplace=True)
        print("Var Frame", self._df_variables)

    def read_json_file(self) -> typing.List:
        """
        Reads the first json file in the path self.filePath

        Parameters:
              void
        Returns:
              data: the contents of the json file

        """
        #try:
            #read_files = glob.glob(os.path.join(self.files_path, "*.json"))
            #if not read_files:
                #raise ValueError('No .json file found in the entered path!')
        with open(self.file_path) as f:
            data = json.load(f)
            return data
        #except ValueError as err:
            #print(err.args)

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
        columns_header.remove(self.time_key)
        self.sorter = columns_header

    def compute_row_delta_sigle_samples_frame(self, sample_frame: pd.DataFrame, time_header_label: str,
                                              columns_header: typing.List, shifted_cols_header: typing.List) \
            -> pd.DataFrame:
        """
        Computes the difference between each value present in th time column.
        Copies and shift by one position up all the values present in the remaining columns.
        Parameters:
            sample_frame: the traj to be processed
            time_header_label: the label for the times
            columns_header: the original header of sample_frame
            shifted_cols_header: a copy of columns_header with changed names of the contents
        Returns:
            sample_frame: the processed dataframe

        """
        sample_frame[time_header_label] = sample_frame[time_header_label].diff().shift(-1)
        shifted_cols = sample_frame[columns_header].shift(-1).fillna(0).astype('int32')
        #print(shifted_cols)
        shifted_cols.columns = shifted_cols_header
        sample_frame = sample_frame.assign(**shifted_cols)
        sample_frame.drop(sample_frame.tail(1).index, inplace=True)
        return sample_frame

    def compute_row_delta_in_all_samples_frames(self, time_header_label: str):
        """
        Calls the method compute_row_delta_sigle_samples_frame on every dataframe present in the list self.df_samples_list.
        Concatenates the result in the dataframe concatanated_samples

        Parameters:
            time_header_label: the label of the time column
        Returns:
            void
        """
        shifted_cols_header = [s + "S" for s in self.sorter]
        compute_row_delta = self.compute_row_delta_sigle_samples_frame
        self.df_samples_list = [compute_row_delta(sample, time_header_label, self.sorter, shifted_cols_header)
                                for sample in self.df_samples_list]
        self._concatenated_samples = pd.concat(self.df_samples_list)
        complete_header = self.sorter[:]
        complete_header.insert(0,'Time')
        complete_header.extend(shifted_cols_header)
        #print("Complete Header", complete_header)
        self._concatenated_samples = self._concatenated_samples[complete_header]
        #print("Concat Samples",self._concatenated_samples)

    def build_list_of_samples_array(self, data_frame: pd.DataFrame) -> typing.List:
        """
        Builds a List containing the columns of dataframe and converts them to a numpy array.
        Parameters:
            :data_frame: the dataframe from which the columns have to be extracted and converted
        Returns:
            :columns_list: the resulting list of numpy arrays
        """
        columns_list = [data_frame[column].to_numpy() for column in data_frame]
        #for column in data_frame:
            #columns_list.append(data_frame[column].to_numpy())
        return columns_list

    def clear_concatenated_frame(self):
        """
        Removes all values in the dataframe concatenated_samples
        Parameters:
            void
        Returns:
            void
         """
        self._concatenated_samples = self._concatenated_samples.iloc[0:0]

    def clear_data_frame_list(self):
        """
        Removes all values present in the dataframes in the list df_samples_list
        """
        for indx in range(len(self.df_samples_list)):  # Le singole traj non servono più #TODO usare list comprens
            self.df_samples_list[indx] = self.df_samples_list[indx].iloc[0:0]

    def import_sampled_cims(self, raw_data: typing.List, indx: int, cims_key: str) -> typing.Dict:
        cims_for_all_vars = {}
        for var in raw_data[indx][cims_key]:
            sampled_cims_list = []
            cims_for_all_vars[var] = sampled_cims_list
            for p_comb in raw_data[indx][cims_key][var]:
                cims_for_all_vars[var].append(pd.DataFrame(raw_data[indx][cims_key][var][p_comb]).to_numpy())
        return cims_for_all_vars

    @property
    def concatenated_samples(self):
        return self._concatenated_samples

    @property
    def variables(self):
        return self._df_variables

    @property
    def structure(self):
        return self._df_structure




