import os
import glob
import pandas as pd
import json
import numpy as np
from abstract_importer import AbstractImporter


class JsonImporter(AbstractImporter):

    def __init__(self, files_path):
        self.df_samples_list = []
        self.df_structure = pd.DataFrame()
        self.df_variables = pd.DataFrame()
        super(JsonImporter, self).__init__(files_path)

    def import_data(self):
        raw_data = self.read_json_file()
        self.import_trajectories(raw_data)
        self.import_structure(raw_data)
        self.import_variables(raw_data)

    def import_trajectories(self, raw_data):
        self.normalize_trajectories(raw_data, 0, 'samples')

    def import_structure(self, raw_data):
        self.df_structure = self.one_level_normalizing(raw_data, 0, 'dyn.str')

    def import_variables(self, raw_data):
        self.df_variables = self.one_level_normalizing(raw_data, 0, 'variables')

    def read_json_file(self):
        try:
            read_files = glob.glob(os.path.join(self.files_path, "*.json"))
            for file_name in read_files:
                with open(file_name) as f:
                    data = json.load(f)
                    return data
        except ValueError as err:
            print(err.args)

    def one_level_normalizing(self, raw_data, indx, variables_key):
        return pd.json_normalize(raw_data[indx][variables_key])

    def normalize_trajectories(self, raw_data, indx, trajectories_key):
        for sample_indx, sample in enumerate(raw_data[indx][trajectories_key]):
            self.df_samples_list.append(pd.json_normalize(raw_data[indx][trajectories_key][sample_indx]))

    def build_list_of_samples_array(self, data_frame):
        columns_list = []
        for column in data_frame:
            columns_list.append(data_frame[column].to_numpy())
        return columns_list

    def clear_data_frames(self):
        """
        Rimuove tutti i valori contenuti nei data_frames presenti in df_list
        Parameters:
            void
        Returns:
            void
         """
        for indx in range(len(self.df_samples_list)):
            #data_frame = data_frame.iloc[0:0]
            self.df_samples_list[indx] = self.df_samples_list[indx].iloc[0:0]


"""ij = JsonImporter("../data")
ij.import_data()
#print(ij.df_samples_list[7])
print(ij.df_structure)
print(ij.df_variables)
print((ij.build_list_of_samples_array(0)[1].size))"""
