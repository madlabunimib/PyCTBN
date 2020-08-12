import sys
sys.path.append("/Users/Zalum/Desktop/Tesi/CTBN_Project/main_package/classes/")
import unittest
import os
import glob
import numpy as np
import pandas as pd
import json_importer as ji

from line_profiler import LineProfiler

import os
import json


class TestJsonImporter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.read_files = glob.glob(os.path.join('../data', "*.json"))

    def test_init(self):
        j1 = ji.JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        self.assertEqual(j1.samples_label, 'samples')
        self.assertEqual(j1.structure_label, 'dyn.str')
        self.assertEqual(j1.variables_label, 'variables')
        self.assertEqual(j1.time_key, 'Time')
        self.assertEqual(j1.variables_key, 'Name')
        self.assertEqual(j1.file_path, self.read_files[0])
        self.assertFalse(j1.df_samples_list)
        self.assertTrue(j1.variables.empty)
        self.assertTrue(j1.structure.empty)
        self.assertFalse(j1.concatenated_samples)
        self.assertFalse(j1.sorter)

    def test_read_json_file_found(self):
        data_set = {"key1": [1, 2, 3], "key2": [4, 5, 6]}
        with open('data.json', 'w') as f:
            json.dump(data_set, f)
        path = os.getcwd()
        path = path + '/data.json'
        j1 = ji.JsonImporter(path, '', '', '', '', '')
        imported_data = j1.read_json_file()
        self.assertTrue(self.ordered(data_set) == self.ordered(imported_data))
        os.remove('data.json')

    def test_read_json_file_not_found(self):
        path = os.getcwd()
        path = path + '/data.json'
        j1 = ji.JsonImporter(path, '', '', '', '', '')
        self.assertRaises(FileNotFoundError, j1.read_json_file)

    def test_normalize_trajectories(self):
        j1 = ji.JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        raw_data = j1.read_json_file()
        #print(raw_data)
        j1.normalize_trajectories(raw_data, 0, j1.samples_label)
        self.assertEqual(len(j1.df_samples_list), len(raw_data[0][j1.samples_label]))
        self.assertEqual(list(j1.df_samples_list[0].columns.values)[1:], j1.sorter)

    def test_normalize_trajectories_wrong_indx(self):
        j1 = ji.JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        raw_data = j1.read_json_file()
        self.assertRaises(IndexError, j1.normalize_trajectories, raw_data, 474, j1.samples_label)

    def test_normalize_trajectories_wrong_key(self):
        j1 = ji.JsonImporter(self.read_files[0], 'sample', 'dyn.str', 'variables', 'Time', 'Name')
        raw_data = j1.read_json_file()
        self.assertRaises(KeyError, j1.normalize_trajectories, raw_data, 0, j1.samples_label)

    def test_compute_row_delta_single_samples_frame(self):
        j1 = ji.JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        raw_data = j1.read_json_file()
        j1.normalize_trajectories(raw_data, 0, j1.samples_label)
        sample_frame = j1.df_samples_list[0]
        columns_header = list(sample_frame.columns.values)
        shifted_cols_header = [s + "S" for s in columns_header[1:]]
        new_sample_frame = j1.compute_row_delta_sigle_samples_frame(sample_frame, j1.time_key, columns_header[1:],
                                                                    shifted_cols_header)
        self.assertEqual(len(list(sample_frame.columns.values)) + len(shifted_cols_header),
                         len(list(new_sample_frame.columns.values)))
        self.assertEqual(sample_frame.shape[0] - 1, new_sample_frame.shape[0])

    def test_compute_row_delta_in_all_frames(self):
        j1 = ji.JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        raw_data = j1.read_json_file()
        j1.import_trajectories(raw_data)
        j1.compute_row_delta_in_all_samples_frames(j1.time_key)
        self.assertEqual(list(j1.df_samples_list[0].columns.values), list(j1.concatenated_samples.columns.values))
        self.assertEqual(list(j1.concatenated_samples.columns.values)[0], j1.time_key)

    def test_clear_data_frame_list(self):
        j1 = ji.JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        raw_data = j1.read_json_file()
        j1.import_trajectories(raw_data)
        j1.compute_row_delta_in_all_samples_frames(j1.time_key)
        j1.clear_data_frame_list()
        for df in j1.df_samples_list:
            self.assertTrue(df.empty)

    def test_build_list_of_samples_array(self):
        data_set = {"key1": [1, 2, 3], "key2": [4.1, 5.2, 6.3]}
        with open('data.json', 'w') as f:
            json.dump(data_set, f)
        path = os.getcwd()
        path = path + '/data.json'
        j1 = ji.JsonImporter(path, '', '', '', '', '')
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
        j1 = ji.JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        sorter = ['X', 'Y', 'Z']
        raw_data = [{'variables':{"Name": ['Z', 'Y', 'X'], "value": [3, 3, 3]}}]
        j1.import_variables(raw_data, sorter)
        self.assertEqual(list(j1.variables[j1.variables_key]), sorter)

    def test_import_data(self):
        j1 = ji.JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        lp = LineProfiler()

        lp_wrapper = lp(j1.import_data)
        lp_wrapper()
        lp.print_stats()
        #j1.import_data()
        self.assertEqual(list(j1.variables[j1.variables_key]),
                         list(j1.concatenated_samples.columns.values[1:len(j1.variables[j1.variables_key]) + 1]))
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
