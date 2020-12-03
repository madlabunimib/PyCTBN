
import unittest
import os
import glob
import numpy as np
import pandas as pd
import json

from ..PyCTBN.json_importer import JsonImporter


class TestJsonImporter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.read_files = glob.glob(os.path.join('./data', "*.json"))
        #print(os.path.join('../data'))

    def test_init(self):
        j1 = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)
        self.assertEqual(j1._samples_label, 'samples')
        self.assertEqual(j1._structure_label, 'dyn.str')
        self.assertEqual(j1._variables_label, 'variables')
        self.assertEqual(j1._time_key, 'Time')
        self.assertEqual(j1._variables_key, 'Name')
        self.assertEqual(j1._file_path, self.read_files[0])
        self.assertIsNone(j1._df_samples_list)
        self.assertIsNone(j1.variables)
        self.assertIsNone(j1.structure)
        self.assertIsNone(j1.concatenated_samples)
        self.assertIsNone(j1.sorter)

    def test_read_json_file_found(self):
        data_set = {"key1": [1, 2, 3], "key2": [4, 5, 6]}
        with open('data.json', 'w') as f:
            json.dump(data_set, f)
        path = os.getcwd()
        path = path + '/data.json'
        j1 = JsonImporter(path, '', '', '', '', '', 0)
        imported_data = j1.read_json_file()
        self.assertTrue(self.ordered(data_set) == self.ordered(imported_data))
        os.remove('data.json')

    def test_read_json_file_not_found(self):
        path = os.getcwd()
        path = path + '/data.json'
        j1 = JsonImporter(path, '', '', '', '', '', 0)
        self.assertRaises(FileNotFoundError, j1.read_json_file)

    def test_normalize_trajectories(self):
        j1 = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)
        raw_data = j1.read_json_file()
        #print(raw_data)
        df_samples_list = j1.normalize_trajectories(raw_data, 0, j1._samples_label)
        self.assertEqual(len(df_samples_list), len(raw_data[0][j1._samples_label]))
        #self.assertEqual(list(j1._df_samples_list[0].columns.values)[1:], j1.sorter)

    def test_normalize_trajectories_wrong_indx(self):
        j1 = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)
        raw_data = j1.read_json_file()
        self.assertRaises(IndexError, j1.normalize_trajectories, raw_data, 474, j1._samples_label)

    def test_normalize_trajectories_wrong_key(self):
        j1 = JsonImporter(self.read_files[0], 'sample', 'dyn.str', 'variables', 'Time', 'Name', 0)
        raw_data = j1.read_json_file()
        self.assertRaises(KeyError, j1.normalize_trajectories, raw_data, 0, j1._samples_label)

    def test_compute_row_delta_single_samples_frame(self):
        j1 = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)
        raw_data = j1.read_json_file()
        j1._df_samples_list = j1.import_trajectories(raw_data)
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
        j1 = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)
        raw_data = j1.read_json_file()
        j1._df_samples_list = j1.import_trajectories(raw_data)
        j1._sorter = j1.build_sorter(j1._df_samples_list[0])
        j1.compute_row_delta_in_all_samples_frames(j1._df_samples_list)
        self.assertEqual(list(j1._df_samples_list[0].columns.values),
                         list(j1.concatenated_samples.columns.values)[:len(list(j1._df_samples_list[0].columns.values))])
        self.assertEqual(list(j1.concatenated_samples.columns.values)[0], j1._time_key)

    def test_clear_data_frame_list(self):
        j1 = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)
        raw_data = j1.read_json_file()
        j1._df_samples_list = j1.import_trajectories(raw_data)
        j1._sorter = j1.build_sorter(j1._df_samples_list[0])
        j1.compute_row_delta_in_all_samples_frames(j1._df_samples_list)
        j1.clear_data_frame_list()
        for df in j1._df_samples_list:
            self.assertTrue(df.empty)

    def test_clear_concatenated_frame(self):
        j1 = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)
        j1.import_data()
        j1.clear_concatenated_frame()
        self.assertTrue(j1.concatenated_samples.empty)

    def test_build_list_of_samples_array(self):
        data_set = {"key1": [1, 2, 3], "key2": [4.1, 5.2, 6.3]}
        with open('data.json', 'w') as f:
            json.dump(data_set, f)
        path = os.getcwd()
        path = path + '/data.json'
        j1 = JsonImporter(path, '', '', '', '', '', 0)
        raw_data = j1.read_json_file()
        frame = pd.DataFrame(raw_data)
        col_list = j1.build_list_of_samples_array(frame)
        forced_list = []
        for key in data_set:
            forced_list.append(np.array(data_set[key]))
        for a1, a2 in zip(col_list, forced_list):
            self.assertTrue(np.array_equal(a1, a2))
        os.remove('data.json')

    def test_import_variables(self):
        j1 = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)
        sorter = ['X', 'Y', 'Z']
        raw_data = [{'variables':{"Name": ['X', 'Y', 'Z'], "value": [3, 3, 3]}}]
        df_var = j1.import_variables(raw_data)
        self.assertEqual(list(df_var[j1._variables_key]), sorter)

    def test_import_structure(self):
        j1 = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)
        raw_data = [{"dyn.str":[{"From":"X","To":"Z"},{"From":"Y","To":"Z"},{"From":"Z","To":"Y"}]}]
        df_struct = j1.import_structure(raw_data)
        #print(raw_data[0]['dyn.str'][0].items())
        self.assertIsInstance(df_struct, pd.DataFrame)

    def test_import_sampled_cims(self):
        j1 = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)
        raw_data = j1.read_json_file()
        j1._df_samples_list = j1.import_trajectories(raw_data)
        j1._sorter = j1.build_sorter(j1._df_samples_list[0])
        cims = j1.import_sampled_cims(raw_data, 0, 'dyn.cims')
        #j1.import_variables(raw_data, j1.sorter)
        self.assertEqual(list(cims.keys()), j1.sorter)

    def test_import_data(self):
        j1 = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 2)
        j1.import_data()
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