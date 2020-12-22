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
import copy

import utility.cache as ch
import structure_graph.sample_path as sp
import estimators.structure_score_based_estimator as se
import utility.json_importer as ji

from multiprocessing import set_start_method




class TestStructureScoreBasedEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #cls.read_files = glob.glob(os.path.join('../../data', "*.json"))
        cls.importer = ji.JsonImporter("/home/alessandro/Documents/ctbn_cba/data/networks_and_trajectories_ternary_data_15.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.s1 = sp.SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()



    def test_structure(self):
        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))

        set_start_method("spawn")
        se1 = se.StructureScoreBasedEstimator(self.s1)
        edges = se1.estimate_structure(
                                    max_parents = None,
                                    iterations_number = 100,
                                    patience = 35,
                                    tabu_length = 15,
                                    tabu_rules_duration = 15,
                                    optimizer = 'tabu',
                                    disable_multiprocessing=False
                                    )
        

        self.assertEqual(edges, true_edges)
        


if __name__ == '__main__':
    unittest.main()

