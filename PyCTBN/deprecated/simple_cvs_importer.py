import pandas as pd
import glob
import os
import typing

from PyCTBN.classes.abstract_importer import AbstractImporter
from PyCTBN.classes.sample_path import SamplePath


class CSVImporter(AbstractImporter):

    def __init__(self, file_path):
        self._df_samples_list = None
        super(CSVImporter, self).__init__(file_path)

    def import_data(self):
        self.read_csv_file()
        self._sorter = self.build_sorter(self._df_samples_list[0])
        self.import_variables()
        self.import_structure()
        self.compute_row_delta_in_all_samples_frames(self._df_samples_list)

    def read_csv_file(self):
        df = pd.read_csv(self._file_path)
        df.drop(df.columns[[0]], axis=1, inplace=True)
        self._df_samples_list = [df]

    def import_variables(self):
        values_list = [3 for var in self._sorter]
        # initialize dict of lists
        data = {'Name':self._sorter, 'Value':values_list}
        # Create the pandas DataFrame
        self._df_variables = pd.DataFrame(data)

    def build_sorter(self, sample_frame: pd.DataFrame) -> typing.List:
        return list(sample_frame.columns)[1:]

    def import_structure(self):
        data = {'From':['X','Y','Z'], 'To':['Z','Z','Y']}
        self._df_structure = pd.DataFrame(data)

    def dataset_id(self) -> object:
        pass


def main():
    read_files = glob.glob(os.path.join('../../data', "*.csv"))
    print(read_files[0])
    csvimp = CSVImporter(read_files[0])
    s1 = SamplePath(csvimp)
    s1.build_trajectories()
    s1.build_structure()
    print(s1.structure)
    print(s1.trajectories)


if __name__ == "__main__":
    main()

