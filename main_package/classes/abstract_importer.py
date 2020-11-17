from abc import ABC, abstractmethod
import pandas as pd
import typing


class AbstractImporter(ABC):
    """
    Interface that exposes all the necessary methods to import the trajectories and the net structure.

    :file_path: the file path
    :_concatenated_samples: the concatenation of all the processed trajectories
    :df_structure: Dataframe containing the structure of the network (edges)
    :df_variables: Dataframe containing the nodes cardinalities
    :df_concatenated_samples: the concatenation and processing of all the trajectories present in the list df_samples list
    :sorter: the columns header(excluding the time column) of the Dataframe concatenated_samples
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._df_variables = None
        self._df_structure = None
        self._concatenated_samples = None
        self._sorter = None
        super().__init__()

    """
    @abstractmethod
    def import_trajectories(self, raw_data):
        pass

    @abstractmethod
    def import_structure(self, raw_data):
        pass
    """

    @abstractmethod
    def import_data(self):
        """
        Imports and prepares all data present needed for susequent computation.
        Parameters:
            void
        Returns:
            void
        POSTCONDITION: the class members self._df_variables and self._df_structure HAVE to be properly constructed
        as Pandas Dataframes
        """
        pass

    def compute_row_delta_sigle_samples_frame(self, sample_frame: pd.DataFrame,
                                              columns_header: typing.List, shifted_cols_header: typing.List) \
            -> pd.DataFrame:
        """
        Computes the difference between each value present in th time column.
        Copies and shift by one position up all the values present in the remaining columns.
        PREREQUISITE: the Dataframe in input has to follow the column structure of this header:
        [Time|Variable values], so it is assumed TIME is ALWAYS the FIRST column.
        Parameters:
            sample_frame: the traj to be processed
            time_header_label: the label for the times
            columns_header: the original header of sample_frame
            shifted_cols_header: a copy of columns_header with changed names of the contents
        Returns:
            sample_frame: the processed dataframe

        """
        #sample_frame[time_header_label] = sample_frame[time_header_label].diff().shift(-1)
        sample_frame.iloc[:, 0] = sample_frame.iloc[:, 0].diff().shift(-1)
        shifted_cols = sample_frame[columns_header].shift(-1).fillna(0).astype('int32')
        shifted_cols.columns = shifted_cols_header
        sample_frame = sample_frame.assign(**shifted_cols)
        sample_frame.drop(sample_frame.tail(1).index, inplace=True)
        return sample_frame

    def compute_row_delta_in_all_samples_frames(self, df_samples_list: typing.List):
        """
        Calls the method compute_row_delta_sigle_samples_frame on every dataframe present in the list df_samples_list.
        Concatenates the result in the dataframe concatanated_samples
        PREREQUISITE: the Dataframe in input has to follow the column structure of this header:
        [Time|Variable values], so it is assumed TIME is ALWAYS the FIRST column.
        The class member self._sorter HAS to be properly INITIALIZED
        Parameters:
            time_header_label: the label of the time column
            df_samples_list: the datframe's list to be processed and concatenated

        Returns:
            void
        """
        shifted_cols_header = [s + "S" for s in self._sorter]
        compute_row_delta = self.compute_row_delta_sigle_samples_frame
        proc_samples_list = [compute_row_delta(sample, self._sorter, shifted_cols_header)
                                for sample in df_samples_list]
        self._concatenated_samples = pd.concat(proc_samples_list)
        complete_header = self._sorter[:]
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

    @property
    def concatenated_samples(self) -> pd.DataFrame:
        return self._concatenated_samples

    @property
    def variables(self) -> pd.DataFrame:
        return self._df_variables

    @property
    def structure(self) -> pd.DataFrame:
        return self._df_structure

    @property
    def sorter(self):
        return self._sorter
