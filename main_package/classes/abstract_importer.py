
import sys
sys.path.append("./classes/")
from abc import ABC, abstractmethod
import pandas as pd
import typing


class AbstractImporter(ABC):
    """Abstract class that exposes all the necessary methods to process the trajectories and the net structure.

    :param file_path: the file path
    :type file_path: str
    :_concatenated_samples: Dataframe containing the concatenation of all the processed trajectories
    :_df_structure: Dataframe containing the structure of the network (edges)
    :_df_variables: Dataframe containing the nodes cardinalities
    :_sorter: A list containing the columns header (excluding the time column) of the `_concatenated_samples`
    """

    def __init__(self, file_path: str):
        """Constructor
        """
        self._file_path = file_path
        self._df_variables = None
        self._df_structure = None
        self._concatenated_samples = None
        self._sorter = None
        super().__init__()

    @abstractmethod
    def import_data(self) -> None:
        """Imports all the trajectories, variables cardinalities, and net edges.

        .. warning::
            The class members ``_df_variables`` and ``_df_structure`` HAVE to be properly constructed
            as Pandas Dataframes with the following structure:
            Header of _df_structure = [From_Node | To_Node]
            Header of _df_variables = [Variable_Label | Variable_Cardinality]
        .. note::
            See :class:``JsonImporter`` for an example of implementation of this method.
        """
        pass

    @abstractmethod
    def build_sorter(self, sample_frame: pd.DataFrame) -> typing.List:
        """Initializes the ``_sorter`` class member from a trajectory dataframe, exctracting the header of the frame
        and keeping ONLY the variables symbolic labels, cutting out the time label in the header.

        :param sample_frame: The dataframe from which extract the header
        :type sample_frame: pandas.DataFrame
        :return: A list containing the processed header.
        :rtype: List
        """
        pass

    def compute_row_delta_sigle_samples_frame(self, sample_frame: pd.DataFrame,
                                              columns_header: typing.List, shifted_cols_header: typing.List) \
            -> pd.DataFrame:
        """Computes the difference between each value present in th time column.
        Copies and shift by one position up all the values present in the remaining columns.

        :param sample_frame: the traj to be processed
        :type sample_frame: pandas.Dataframe
        :param columns_header: the original header of sample_frame
        :type columns_header: List
        :param shifted_cols_header: a copy of columns_header with changed names of the contents
        :type shifted_cols_header: List
        :return: The processed dataframe
        :rtype: pandas.Dataframe

        .. warning::
            the Dataframe ``sample_frame`` has to follow the column structure of this header:
            Header of sample_frame = [Time | Variable values]
        """
        sample_frame.iloc[:, 0] = sample_frame.iloc[:, 0].diff().shift(-1)
        shifted_cols = sample_frame[columns_header].shift(-1).fillna(0).astype('int32')
        shifted_cols.columns = shifted_cols_header
        sample_frame = sample_frame.assign(**shifted_cols)
        sample_frame.drop(sample_frame.tail(1).index, inplace=True)
        return sample_frame

    def compute_row_delta_in_all_samples_frames(self, df_samples_list: typing.List) -> None:
        """Calls the method ``compute_row_delta_sigle_samples_frame`` on every dataframe present in the list
        ``df_samples_list``.
        Concatenates the result in the dataframe ``concatanated_samples``

        :param df_samples_list: the datframe's list to be processed and concatenated
        :type df_samples_list: List

        .. warning::
            The Dataframe sample_frame has to follow the column structure of this header:
            Header of sample_frame = [Time | Variable values]
            The class member self._sorter HAS to be properly INITIALIZED (See class members definition doc)
        .. note::
            After the call of this method the class member ``concatanated_samples`` will contain all processed
            and merged trajectories
        """
        if not self._sorter:
            raise RuntimeError("The class member self._sorter has to be INITIALIZED!")
        shifted_cols_header = [s + "S" for s in self._sorter]
        compute_row_delta = self.compute_row_delta_sigle_samples_frame
        proc_samples_list = [compute_row_delta(sample, self._sorter, shifted_cols_header)
                                for sample in df_samples_list]
        self._concatenated_samples = pd.concat(proc_samples_list)
        complete_header = self._sorter[:]
        complete_header.insert(0,'Time')
        complete_header.extend(shifted_cols_header)
        self._concatenated_samples = self._concatenated_samples[complete_header]

    def build_list_of_samples_array(self, data_frame: pd.DataFrame) -> typing.List:
        """Builds a List containing the columns of data_frame and converts them to a numpy array.

        :param data_frame: the dataframe from which the columns have to be extracted and converted
        :type data_frame: pandas.Dataframe
        :return: the resulting list of numpy arrays
        :rtype: List
        """
        columns_list = [data_frame[column].to_numpy() for column in data_frame]
        return columns_list

    def clear_concatenated_frame(self) -> None:
        """Removes all values in the dataframe concatenated_samples.
         """
        self._concatenated_samples = self._concatenated_samples.iloc[0:0]

    @abstractmethod
    def dataset_id(self) -> object:
        """If the original dataset contains multiple dataset, this method returns a unique id to identify the current
        dataset
        """
        pass

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
    def sorter(self) -> typing.List:
        return self._sorter

    @property
    def file_path(self) -> str:
        return self._file_path
