import sys
sys.path.append("../../classes/")
import glob
import math
import os
import unittest

import networkx as nx
import numpy as np
import psutil
from line_profiler import LineProfiler

import utility.cache as ch
import structure_graph.sample_path as sp
import estimators.structure_score_based_estimator as se
import utility.json_importer as ji



class TestStructureScoreBasedEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #cls.read_files = glob.glob(os.path.join('../../data', "*.json"))
        cls.importer = ji.JsonImporter("../../data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.s1 = sp.SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()



    def test_esecuzione(self):
        se1 = se.StructureScoreBasedEstimator(self.s1)
        se1.estimate_structure(
                            max_parents = 6,
                            iterations_number = 80,
                            patience = None 
                            )

if __name__ == '__main__':
    unittest.main()

    '''

        def test_init(self):
        exp_alfa = 0.1
        chi_alfa = 0.1
        se1 = se.StructureScoreBasedEstimator(self.s1)
        self.assertEqual(self.s1, se1.sample_path)
        self.assertTrue(np.array_equal(se1.nodes, np.array(self.s1.structure.nodes_labels)))
        self.assertTrue(np.array_equal(se1.nodes_indxs, self.s1.structure.nodes_indexes))
        self.assertTrue(np.array_equal(se1.nodes_vals, self.s1.structure.nodes_values))
        self.assertIsInstance(se1.complete_graph, nx.DiGraph)
        self.assertIsInstance(se1.cache, ch.Cache)

    def test_build_complete_graph(self):
        nodes_numb = len(self.s1.structure.nodes_labels)
        se1 = se.StructureScoreBasedEstimator(self.s1)
        cg = se1.build_complete_graph(self.s1.structure.nodes_labels)
        self.assertEqual(len(cg.edges), nodes_numb*(nodes_numb - 1))
        ''for node in self.s1.structure.nodes_labels:
            no_self_loops = self.s1.structure.nodes_labels[:]
            no_self_loops.remove(node)
            for n2 in no_self_loops:
                self.assertIn((node, n2), cg.edges)''
    '''
