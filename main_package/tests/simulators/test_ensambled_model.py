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
import estimators.structure_constraint_based_estimator as se_
import utility.json_importer as ji



class TestHybridMethod(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #cls.read_files = glob.glob(os.path.join('../../data', "*.json"))
        cls.importer = ji.JsonImporter("../../data/networks_and_trajectories_binary_data_04_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.s1 = sp.SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()



    def test_structure(self):
        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))

        s2= copy.deepcopy(self.s1)

        se1 = se.StructureScoreBasedEstimator(self.s1,1,1)
        edges_score = se1.estimate_structure(
                                max_parents = None,
                                iterations_number = 100,
                                patience = 50,
                                tabu_length = 20,
                                tabu_rules_duration = 20,
                                optimizer = 'tabu'
                            )

        se2 = se_.StructureConstraintBasedEstimator(s2, 0.1, 0.1)
        edges_constraint = se2.estimate_structure()
        
        set_list_edges = set.union(edges_constraint,edges_score)

        
        n_added_fake_edges = len(set_list_edges.difference(true_edges))

        n_missing_edges = len(true_edges.difference(set_list_edges))

        n_true_positive = len(true_edges) - n_missing_edges

        precision = n_true_positive / (n_true_positive + n_added_fake_edges)

        recall = n_true_positive / (n_true_positive + n_missing_edges)

        f1_measure = round(2* (precision*recall) / (precision+recall),3)

        # print(f"n archi reali non trovati: {n_missing_edges}")
        # print(f"n archi non reali aggiunti: {n_added_fake_edges}")
        print(true_edges)
        print(set_list_edges)
        print(f"precision: {precision} ")
        print(f"recall: {recall} ")
        print(f"F1: {f1_measure} ")



        self.assertEqual(set_list_edges, true_edges)
        


if __name__ == '__main__':
    unittest.main()

