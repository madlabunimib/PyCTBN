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





class TestTabuSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #cls.read_files = glob.glob(os.path.join('../../data', "*.json"))
        cls.importer = ji.JsonImporter("../../data/networks_and_trajectories_ternary_data_5.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.importer.import_data(0)
        cls.s1 = sp.SamplePath(cls.importer)

        #cls.traj = cls.s1.concatenated_samples

       # print(len(cls.traj))
        cls.s1 = sp.SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()
        cls.s1.clear_memory() 



    def test_structure(self):
        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))

        se1 = se.StructureScoreBasedEstimator(self.s1)
        edges = se1.estimate_structure(
                            max_parents = None,
                            iterations_number = 100,
                            patience = 20,
                            tabu_length = 10,
                            tabu_rules_duration = 10,
                            optimizer = 'tabu',
                            disable_multiprocessing=False
                            )
        

        self.assertEqual(edges, true_edges)
        


if __name__ == '__main__':
    unittest.main()

