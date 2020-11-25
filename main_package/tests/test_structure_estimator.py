import sys
sys.path.append("../classes/")
import glob
import math
import os
import unittest

import networkx as nx
import numpy as np
import psutil
from line_profiler import LineProfiler

import cache as ch
import sample_path as sp
import structure_estimator as se
import json_importer as ji


class TestStructureEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.read_files = glob.glob(os.path.join('../data', "*.json"))
        cls.importer = ji.JsonImporter(cls.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 3)
        cls.s1 = sp.SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()

    def test_init(self):
        exp_alfa = 0.1
        chi_alfa = 0.1
        se1 = se.StructureEstimator(self.s1, exp_alfa, chi_alfa)
        self.assertEqual(self.s1, se1._sample_path)
        self.assertTrue(np.array_equal(se1._nodes, np.array(self.s1.structure.nodes_labels)))
        self.assertTrue(np.array_equal(se1._nodes_indxs, self.s1.structure.nodes_indexes))
        self.assertTrue(np.array_equal(se1._nodes_vals, self.s1.structure.nodes_values))
        self.assertEqual(se1._exp_test_sign, exp_alfa)
        self.assertEqual(se1._chi_test_alfa, chi_alfa)
        self.assertIsInstance(se1._complete_graph, nx.DiGraph)
        self.assertIsInstance(se1._cache, ch.Cache)

    def test_build_complete_graph(self):
        exp_alfa = 0.1
        chi_alfa = 0.1
        nodes_numb = len(self.s1.structure.nodes_labels)
        se1 = se.StructureEstimator(self.s1, exp_alfa, chi_alfa)
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
        se1 = se.StructureEstimator(self.s1, exp_alfa, chi_alfa)

        for node in self.s1.structure.nodes_labels:
            for b in range(nodes_numb):
                sets = se1.generate_possible_sub_sets_of_size(self.s1.structure.nodes_labels, b, node)
                sets2 = se1.generate_possible_sub_sets_of_size(self.s1.structure.nodes_labels, b, node)
                self.assertEqual(len(list(sets)), math.floor(math.factorial(nodes_numb - 1) /
                                                             (math.factorial(b)*math.factorial(nodes_numb -1 - b))))
                for sset in sets2:
                    self.assertFalse(node in sset)

    def test_time(self):
        se1 = se.StructureEstimator(self.s1, 0.1, 0.1)
        lp = LineProfiler()
        lp.add_function(se1.complete_test)
        lp.add_function(se1.one_iteration_of_CTPC_algorithm)
        lp.add_function(se1.independence_test)
        lp_wrapper = lp(se1.ctpc_algorithm)
        lp_wrapper()
        lp.print_stats()
        print(se1._complete_graph.edges)
        print(self.s1.structure.edges)
        for ed in self.s1.structure.edges:
            self.assertIn(tuple(ed), se1._complete_graph.edges)
        tuples_edges = [tuple(rec) for rec in self.s1.structure.edges]
        spurious_edges = []
        for ed in se1._complete_graph.edges:
            if not(ed in tuples_edges):
                spurious_edges.append(ed)
        print("Spurious Edges:",spurious_edges)
        print("Adj Matrix:", nx.adj_matrix(se1._complete_graph).toarray().astype(bool))
        #se1.save_results()

    def test_memory(self):
        se1 = se.StructureEstimator(self.s1, 0.1, 0.1)
        se1.ctpc_algorithm()
        current_process = psutil.Process(os.getpid())
        mem = current_process.memory_info().rss
        print("Average Memory Usage in MB:", mem / 10**6)


if __name__ == '__main__':
    unittest.main()
