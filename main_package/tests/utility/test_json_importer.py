
import unittest
import os
import glob
import numpy as np
import pandas as pd
from ...classes.utility.json_importer import JsonImporter

import json



class TestJsonImporter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.read_files = glob.glob(os.path.join('./main_package/data', "*.json"))

    def test_init(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        self.assertEqual(j1._samples_label, 'samples')
        self.assertEqual(j1._structure_label, 'dyn.str')
        self.assertEqual(j1._variables_label, 'variables')
        self.assertEqual(j1._time_key, 'Time')
        self.assertEqual(j1._variables_key, 'Name')
        self.assertEqual(j1._file_path, "./main_package/data/networks_and_trajectories_binary_data_01_3.json")
        self.assertIsNone(j1._df_samples_list)
        self.assertIsNone(j1.variables)
        self.assertIsNone(j1.structure)
        self.assertEqual(j1.concatenated_samples,[])
        self.assertIsNone(j1.sorter)
        self.assertIsNone(j1._array_indx)
        self.assertIsInstance(j1._raw_data, list)

    def test_read_json_file_found(self):
        data_set = {"key1": [1, 2, 3], "key2": [4, 5, 6]}
        with open('data.json', 'w') as f:
            json.dump(data_set, f)
        path = os.getcwd()
        path = path + '/data.json'
        j1 = JsonImporter(path, '', '', '', '', '')
        self.assertTrue(self.ordered(data_set) == self.ordered(j1._raw_data))
        os.remove('data.json')

    def test_read_json_file_not_found(self):
        path = os.getcwd()
        path = path + '/data.json'
        self.assertRaises(FileNotFoundError, JsonImporter, path, '', '', '', '', '')

    def test_build_sorter(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        df_samples_list = j1.normalize_trajectories(j1._raw_data, 0, j1._samples_label)
        sorter = j1.build_sorter(df_samples_list[0])
        self.assertListEqual(sorter, list(df_samples_list[0].columns.values)[1:])

    def test_normalize_trajectories(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        df_samples_list = j1.normalize_trajectories(j1._raw_data, 0, j1._samples_label)
        self.assertEqual(len(df_samples_list), len(j1._raw_data[0][j1._samples_label]))

    def test_normalize_trajectories_wrong_indx(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        self.assertRaises(IndexError, j1.normalize_trajectories, j1._raw_data, 474, j1._samples_label)

    def test_normalize_trajectories_wrong_key(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'sample', 'dyn.str', 'variables', 'Time', 'Name')
        self.assertRaises(KeyError, j1.normalize_trajectories, j1._raw_data, 0, j1._samples_label)

    def test_compute_row_delta_single_samples_frame(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        j1._array_indx = 0
        j1._df_samples_list = j1.import_trajectories(j1._raw_data)
        sample_frame = j1._df_samples_list[0]
        original_copy = sample_frame.copy()
        columns_header = list(sample_frame.columns.values)
        shifted_cols_header = [s + "S" for s in columns_header[1:]]
        new_sample_frame = j1.compute_row_delta_sigle_samples_frame(sample_frame, columns_header[1:],
                                                                    shifted_cols_header)
        self.assertEqual(len(list(sample_frame.columns.values)) + len(shifted_cols_header),
                         len(list(new_sample_frame.columns.values)))
        self.assertEqual(sample_frame.shape[0] - 1, new_sample_frame.shape[0])
        for indx, row in new_sample_frame.iterrows():
            self.assertAlmostEqual(row['Time'],
                                   original_copy.iloc[indx + 1]['Time'] - original_copy.iloc[indx]['Time'])
        for indx, row in new_sample_frame.iterrows():
            np.array_equal(np.array(row[columns_header[1:]],dtype=int),
                  np.array(original_copy.iloc[indx][columns_header[1:]],dtype=int))
            np.array_equal(np.array(row[shifted_cols_header], dtype=int),
                           np.array(original_copy.iloc[indx + 1][columns_header[1:]], dtype=int))

    def test_compute_row_delta_in_all_frames(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        j1._array_indx = 0
        j1._df_samples_list = j1.import_trajectories(j1._raw_data)
        j1._sorter = j1.build_sorter(j1._df_samples_list[0])
        j1.compute_row_delta_in_all_samples_frames(j1._df_samples_list)
        self.assertEqual(list(j1._df_samples_list[0].columns.values),
                         list(j1.concatenated_samples.columns.values)[:len(list(j1._df_samples_list[0].columns.values))])
        self.assertEqual(list(j1.concatenated_samples.columns.values)[0], j1._time_key)

    def test_compute_row_delta_in_all_frames_not_init_sorter(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        j1._array_indx = 0
        j1._df_samples_list = j1.import_trajectories(j1._raw_data)
        self.assertRaises(RuntimeError, j1.compute_row_delta_in_all_samples_frames, j1._df_samples_list)

    def test_clear_data_frame_list(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        j1._array_indx = 0
        j1._df_samples_list = j1.import_trajectories(j1._raw_data)
        j1._sorter = j1.build_sorter(j1._df_samples_list[0])
        j1.compute_row_delta_in_all_samples_frames(j1._df_samples_list)
        j1.clear_data_frame_list()
        for df in j1._df_samples_list:
            self.assertTrue(df.empty)

    def test_clear_concatenated_frame(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        j1.import_data(0)
        j1.clear_concatenated_frame()
        self.assertTrue(j1.concatenated_samples.empty)

    def test_import_variables(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        sorter = ['X', 'Y', 'Z']
        raw_data = [{'variables':{"Name": ['X', 'Y', 'Z'], "value": [3, 3, 3]}}]
        j1._array_indx = 0
        df_var = j1.import_variables(raw_data)
        self.assertEqual(list(df_var[j1._variables_key]), sorter)

    def test_import_structure(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        raw_data = [{"dyn.str":[{"From":"X","To":"Z"},{"From":"Y","To":"Z"},{"From":"Z","To":"Y"}]}]
        j1._array_indx = 0
        df_struct = j1.import_structure(raw_data)
        self.assertIsInstance(df_struct, pd.DataFrame)

    def test_import_sampled_cims(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        raw_data = j1.read_json_file()
        j1._array_indx = 0
        j1._df_samples_list = j1.import_trajectories(raw_data)
        j1._sorter = j1.build_sorter(j1._df_samples_list[0])
        cims = j1.import_sampled_cims(raw_data, 0, 'dyn.cims')
        self.assertEqual(list(cims.keys()), j1.sorter)

    def test_dataset_id(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        array_indx = 0
        j1.import_data(array_indx)
        self.assertEqual(array_indx, j1.dataset_id())

    def test_file_path(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        self.assertEqual(j1.file_path, "./main_package/data/networks_and_trajectories_binary_data_01_3.json")

    def test_import_data(self):
        j1 = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        j1.import_data(0)
        self.assertEqual(list(j1.variables[j1._variables_key]),
                         list(j1.concatenated_samples.columns.values[1:len(j1.variables[j1._variables_key]) + 1]))
        print(j1.variables)
        print(j1.structure)
        print(j1.concatenated_samples)

    def ordered(self, obj):
        if isinstance(obj, dict):
            return sorted((k, self.ordered(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return sorted(self.ordered(x) for x in obj)
        else:
            return obj


if __name__ == '__main__':
    unittest.main()
