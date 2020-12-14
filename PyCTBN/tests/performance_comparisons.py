import unittest
import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
import timeit
import psutil

from ..PyCTBN.sample_path import SamplePath
from ..PyCTBN.structure_estimator import StructureEstimator
from ..PyCTBN.json_importer import JsonImporter
from .original_ctpc_algorithm import OriginalCTPCAlgorithm


class PerformanceComparisons(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.read_files = glob.glob(os.path.join('./data', "*.json"))
        cls.datasets_numb = None
        cls.s1 = None
        cls.importer = None
        cls.original_algo = None
        cls.real_net_graph = None
        cls.original_times = []
        cls.optimized_times = []
        cls.results = []
        cls.memory_usages = []

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
                try:
                    self.assertTrue(np.array_equal(original_res, opt_res))
                except AssertionError:
                    self.real_net_graph = self.real_net_structure_builder()
                    original_algo_net_graph = self.build_graph_from_adj_matrix(original_res, self.s1.structure.nodes_labels)
                    original_spurius_edges = self.compute_edges_difference(self.real_net_graph, original_algo_net_graph)
                    opt_spurius_edges = self.compute_edges_difference(self.real_net_graph, se1._complete_graph)
                    self.assertLessEqual(opt_spurius_edges, original_spurius_edges)
                    continue
            self.save_datas(self.original_times, self.optimized_times)
            self.original_times[:] = []
            self.optimized_times[:] = []
    """
    def test_memory_usage(self):
        for file_path in self.read_files:
            self.importer = JsonImporter(file_path, 'samples', 'dyn.str', 'variables', 'Time', 'Name')
            self.aux_build_importer(0)
            se1 = StructureEstimator(self.s1, 0.1, 0.1)
            se1.ctpc_algorithm()
            current_process = psutil.Process(os.getpid())
            mem = current_process.memory_info().rss
            self.memory_usages.append((mem / 10 ** 6))
        self.save_memory_usage_data(self.memory_usages)
    """

    def aux_build_importer(self, indx):
        self.importer.import_data(indx)
        self.s1 = SamplePath(self.importer)
        self.s1.build_trajectories()
        self.s1.build_structure()

    def aux_build_original_algo(self, indx):
        self.original_algo.import_data(indx)
        self.original_algo.prepare_trajectories(self.original_algo.df_samples_list, self.original_algo.variables)

    def save_datas(self, original_list, opt_list):
        if not os.path.exists('results'):
            os.makedirs('results')
        df_results = pd.DataFrame({'orginal_execution_time': original_list, 'optimized_execution_time': opt_list})
        name = self.importer.file_path.rsplit('/', 1)[-1]
        name = name.split('.', 1)[0]
        name = 'execution_times_' + name + '.csv'
        path = os.path.abspath('./results/')
        file_dest = path + '/' + name
        df_results.to_csv(file_dest, index=False)

    def save_memory_usage_data(self, data):
        if not os.path.exists('memory_results'):
            os.makedirs('memory_results')
        df_results = pd.DataFrame({'memory_usage': data})
        name = self.importer.file_path.rsplit('/', 1)[-1]
        name = name.split('.', 1)[0]
        name = 'memory_usage_' + name + '.csv'
        path = os.path.abspath('./memory_results/')
        file_dest = path + '/' + name
        df_results.to_csv(file_dest, index=False)


    def real_net_structure_builder(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.s1.structure.nodes_labels)
        graph.add_edges_from(self.s1.structure.edges)
        return graph

    def build_graph_from_adj_matrix(self, original_res, nodes_labels):
        graph = nx.from_numpy_matrix(original_res, create_using=nx.DiGraph)
        mapping = dict(zip(graph.nodes, nodes_labels))
        graph = nx.relabel_nodes(graph, mapping)
        return graph

    def compute_edges_difference(self, g1, g2):
        return len(nx.difference(g1, g2))


if __name__ == '__main__':
    unittest.main()
