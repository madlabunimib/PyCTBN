import unittest
import os
import glob
import numpy as np
import pandas as pd
import timeit

from ..PyCTBN.sample_path import SamplePath
from ..PyCTBN.structure_estimator import StructureEstimator
from ..PyCTBN.json_importer import JsonImporter
from PyCTBN.tests.original_ctpc_algorithm import OriginalCTPCAlgorithm


class PerformanceComparisons(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.read_files = glob.glob(os.path.join('./data', "*.json"))
        cls.datasets_numb = None
        cls.s1 = None
        cls.importer = None
        cls.original_algo = None
        cls.original_times = []
        cls.optimized_times = []
        cls.results = []

    def test_time_comparisons(self):
        for file_path in self.read_files:
            self.importer = JsonImporter(file_path, 'samples', 'dyn.str', 'variables', 'Time', 'Name')
            self.original_algo = OriginalCTPCAlgorithm(file_path, 'samples', 'dyn.str', 'variables',
                                                       'Time', 'Name', self.importer._raw_data)
            self.datasets_numb = self.original_algo.datasets_numb()
            print("Testing on file: ", file_path, " with ", self.datasets_numb, " Datasets")
            for indx in range(self.datasets_numb):
                print("Running Test: ", indx)
                self.aux_build_importer(indx)
                self.aux_build_original_algo(indx)
                se1 = StructureEstimator(self.s1, 0.1, 0.1)
                opt_time = timeit.timeit(se1.ctpc_algorithm, number=1)
                original_time = timeit.timeit(self.original_algo.cb_structure_algo, number=1)
                opt_res = se1.adjacency_matrix()
                original_res = self.original_algo.matrix
                self.original_times.append(original_time)
                self.optimized_times.append(opt_time)
                self.results.append((original_res, opt_res))
                self.assertTrue(np.array_equal(original_res, opt_res))
            self.save_datas(self.original_times, self.optimized_times)
            self.original_times[:] = []
            self.optimized_times[:] = []

    def aux_build_importer(self, indx):
        self.importer.import_data(indx)
        self.s1 = SamplePath(self.importer)
        self.s1.build_trajectories()
        self.s1.build_structure()

    def aux_build_original_algo(self, indx):
        self.original_algo.import_data(indx)
        self.original_algo.prepare_trajectories(self.original_algo.df_samples_list, self.original_algo.variables)

    def save_datas(self, original_list, opt_list):
        df_results = pd.DataFrame({'orginal_execution_time': original_list, 'optimized_execution_time': opt_list})
        name = self.importer.file_path.rsplit('/', 1)[-1]
        name = name.split('.', 1)[0]
        name = 'execution_times_' + name + '.csv'
        df_results.to_csv(name, index=False)


if __name__ == '__main__':
    unittest.main()
