
import glob
import math
import os
import unittest
import json
import networkx as nx
import numpy as np
import timeit

from ..PyCTBN.cache import Cache
from ..PyCTBN.sample_path import SamplePath
from ..PyCTBN.structure_estimator import StructureEstimator
from ..PyCTBN.json_importer import JsonImporter


class TestStructureEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.read_files = glob.glob(os.path.join('./data', "*.json"))
        cls.importer = JsonImporter(cls.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.importer.import_data(0)
        cls.s1 = SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()

    def test_init(self):
        exp_alfa = 0.1
        chi_alfa = 0.1
        se1 = StructureEstimator(self.s1, exp_alfa, chi_alfa)
        self.assertEqual(self.s1, se1._sample_path)
        self.assertTrue(np.array_equal(se1._nodes, np.array(self.s1.structure.nodes_labels)))
        self.assertTrue(np.array_equal(se1._nodes_indxs, self.s1.structure.nodes_indexes))
        self.assertTrue(np.array_equal(se1._nodes_vals, self.s1.structure.nodes_values))
        self.assertEqual(se1._exp_test_sign, exp_alfa)
        self.assertEqual(se1._chi_test_alfa, chi_alfa)
        self.assertIsInstance(se1._complete_graph, nx.DiGraph)
        self.assertIsInstance(se1._cache, Cache)

    def test_build_complete_graph(self):
        exp_alfa = 0.1
        chi_alfa = 0.1
        nodes_numb = len(self.s1.structure.nodes_labels)
        se1 = StructureEstimator(self.s1, exp_alfa, chi_alfa)
        cg = se1.build_complete_graph(self.s1.structure.nodes_labels)
        self.assertEqual(len(cg.edges), nodes_numb*(nodes_numb - 1))
        for node in self.s1.structure.nodes_labels:
            no_self_loops = self.s1.structure.nodes_labels[:]
            no_self_loops.remove(node)
            for n2 in no_self_loops:
                self.assertIn((node, n2), cg.edges)

    def test_generate_possible_sub_sets_of_size(self):
        exp_alfa = 0.1
        chi_alfa = 0.1
        nodes_numb = len(self.s1.structure.nodes_labels)
        se1 = StructureEstimator(self.s1, exp_alfa, chi_alfa)

        for node in self.s1.structure.nodes_labels:
            for b in range(nodes_numb):
                sets = StructureEstimator.generate_possible_sub_sets_of_size(self.s1.structure.nodes_labels, b, node)
                sets2 = StructureEstimator.generate_possible_sub_sets_of_size(self.s1.structure.nodes_labels, b, node)
                self.assertEqual(len(list(sets)), math.floor(math.factorial(nodes_numb - 1) /
                                                             (math.factorial(b)*math.factorial(nodes_numb -1 - b))))
                for sset in sets2:
                    self.assertFalse(node in sset)

    def test_time(self):
        se1 = StructureEstimator(self.s1, 0.1, 0.1)
        exec_time = timeit.timeit(se1.ctpc_algorithm, number=1)
        print("Execution Time: ", exec_time)
        for ed in self.s1.structure.edges:
            self.assertIn(tuple(ed), se1._complete_graph.edges)
        tuples_edges = [tuple(rec) for rec in self.s1.structure.edges]
        spurious_edges = []
        for ed in se1._complete_graph.edges:
            if not(ed in tuples_edges):
                spurious_edges.append(ed)
        print("Spurious Edges:",spurious_edges)

    def test_save_results(self):
        se1 = StructureEstimator(self.s1, 0.1, 0.1)
        se1.ctpc_algorithm()
        se1.save_results()
        name = self.s1._importer.file_path.rsplit('/', 1)[-1]
        name = name.split('.', 1)[0]
        name += '_' + str(self.s1._importer.dataset_id())
        name += '.json'
        file_name = 'results_' + name
        with open(file_name) as f:
            js_graph = json.load(f)
            result_graph = nx.json_graph.node_link_graph(js_graph)
            self.assertFalse(nx.difference(se1._complete_graph, result_graph).edges)
            os.remove(file_name)

    def test_adjacency_matrix(self):
        se1 = StructureEstimator(self.s1, 0.1, 0.1)
        se1.ctpc_algorithm()
        adj_matrix = nx.adj_matrix(se1._complete_graph).toarray().astype(bool)
        self.assertTrue(np.array_equal(adj_matrix, se1.adjacency_matrix()))


if __name__ == '__main__':
    unittest.main()
