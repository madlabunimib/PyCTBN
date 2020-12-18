
import glob
import math
import os
import unittest

import networkx as nx
import numpy as np
import psutil
from line_profiler import LineProfiler
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
        cls.importer.import_data(2)
        cls.s1 = SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()

    def test_init(self):
        exp_alfa = 0.1
        chi_alfa = 0.1
        se1 = StructureEstimator(self.s1, exp_alfa, chi_alfa)
        #self.assertEqual(self.s1, se1._sample_path)
        self.assertTrue(np.array_equal(se1._nodes, np.array(self.s1.structure.nodes_labels)))
        self.assertTrue(np.array_equal(se1._nodes_indxs, self.s1.structure.nodes_indexes))
        self.assertTrue(np.array_equal(se1._nodes_vals, self.s1.structure.nodes_values))
        self.assertEqual(se1._exp_test_sign, exp_alfa)
        self.assertEqual(se1._chi_test_alfa, chi_alfa)
        self.assertIsInstance(se1._complete_graph, nx.DiGraph)
        #self.assertIsInstance(se1._cache, Cache)

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

    def test_build_result_graph(self):
        exp_alfa = 0.1
        chi_alfa = 0.1
        nodes_numb = len(self.s1.structure.nodes_labels)
        se1 = StructureEstimator(self.s1, exp_alfa, chi_alfa)
        t1 = se1.build_result_graph(['X','Y','Z'], [[],['X', 'Z'],['X', 'Y']])
        print(t1.edges)

    def test_generate_possible_sub_sets_of_size(self):
        exp_alfa = 0.1
        chi_alfa = 0.1
        nodes_numb = len(self.s1.structure.nodes_labels)
        se1 = StructureEstimator(self.s1, exp_alfa, chi_alfa)

        for node in self.s1.structure.nodes_labels:
            for b in range(nodes_numb):
                sets = se1.generate_possible_sub_sets_of_size(self.s1.structure.nodes_labels, b, node)
                sets2 = se1.generate_possible_sub_sets_of_size(self.s1.structure.nodes_labels, b, node)
                self.assertEqual(len(list(sets)), math.floor(math.factorial(nodes_numb - 1) /
                                                             (math.factorial(b)*math.factorial(nodes_numb -1 - b))))
                for sset in sets2:
                    self.assertFalse(node in sset)

    def test_time(self):
        se1 = StructureEstimator(self.s1, 0.1, 0.1)
        lp = LineProfiler()
        MULTI_PROCESSING = True ###### MODIFICARE QUI SINGLE/MULTI PROCESS
        lp_wrapper = lp(se1.ctpc_algorithm)
        lp_wrapper(MULTI_PROCESSING)
        lp.print_stats()
        #paralell_time = timeit.timeit(se1.ctpc_algorithm, MULTI_PROCESSING, number=1)
        #print("EXEC TIME:", paralell_time)
        print(se1._result_graph.edges)
        #print(self.s1.structure.edges)
        for ed in self.s1.structure.edges:
            self.assertIn(tuple(ed), se1._result_graph.edges)
        tuples_edges = [tuple(rec) for rec in self.s1.structure.edges]
        spurious_edges = []
        for ed in se1._result_graph.edges:
            if not(ed in tuples_edges):
                spurious_edges.append(ed)
        print("Spurious Edges:",spurious_edges)
        print("Adj Matrix:", nx.adj_matrix(se1._result_graph).toarray().astype(bool))
        #se1.save_results()

    """
    def test_memory(self):
        se1 = StructureEstimator(self.s1, 0.1, 0.1)
        se1.ctpc_algorithm()
        current_process = psutil.Process(os.getpid())
        mem = current_process.memory_info().rss
        print("Average Memory Usage in MB:", mem / 10**6)
    """


if __name__ == '__main__':
    unittest.main()
