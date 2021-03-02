
import glob
import math
import os
import unittest

import networkx as nx
import numpy as np
import psutil
from line_profiler import LineProfiler
import copy

from ...PyCTBN.utility.cache import Cache
from ...PyCTBN.structure_graph.sample_path import SamplePath
from ...PyCTBN.estimators.structure_score_based_estimator import StructureScoreBasedEstimator
from ...PyCTBN.utility.json_importer import JsonImporter
from ...PyCTBN.utility.sample_importer import SampleImporter

import json

import pandas as pd



class TestStructureScoreBasedEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open("./PyCTBN/test_data/networks_and_trajectories_binary_data_01_3.json") as f:
            raw_data = json.load(f)

            trajectory_list_raw= raw_data[0]["samples"]

            trajectory_list = [pd.DataFrame(sample) for sample in trajectory_list_raw]

            variables= pd.DataFrame(raw_data[0]["variables"])
            prior_net_structure = pd.DataFrame(raw_data[0]["dyn.str"])


        cls.importer = SampleImporter(
                                        trajectory_list=trajectory_list,
                                        variables=variables,
                                        prior_net_structure=prior_net_structure
                                    )
        
        cls.importer.import_data()
        #cls.s1 = sp.SamplePath(cls.importer)

        #cls.traj = cls.s1.concatenated_samples

       # print(len(cls.traj))
        cls.s1 = SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()

    def test_structure_monoprocesso(self):
        with open("./PyCTBN/test_data/networks_and_trajectories_binary_data_01_3.json") as f:
            raw_data = json.load(f)

            trajectory_list_raw= raw_data[0]["samples"]

            trajectory_list = [pd.DataFrame(sample) for sample in trajectory_list_raw]

            variables= pd.DataFrame(raw_data[0]["variables"])
            prior_net_structure = pd.DataFrame(raw_data[0]["dyn.str"])


        self.importer = SampleImporter(
                                        trajectory_list=trajectory_list,
                                        variables=variables,
                                        prior_net_structure=prior_net_structure
                                    )
        
        self.importer.import_data()
        #cls.s1 = sp.SamplePath(cls.importer)

        #cls.traj = cls.s1.concatenated_samples

       # print(len(cls.traj))
        self.s1 = SamplePath(self.importer)
        self.s1.build_trajectories()
        self.s1.build_structure()
        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))

        
        se1 = StructureScoreBasedEstimator(self.s1)
        edges = se1.estimate_structure(
                                    max_parents = None,
                                    iterations_number = 100,
                                    patience = 35,
                                    tabu_length = 15,
                                    tabu_rules_duration = 15,
                                    optimizer = 'hill',
                                    disable_multiprocessing=True
                                    )
        

        self.assertEqual(edges, true_edges)

    def test_structure_1(self):
        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))

        
        se1 = StructureScoreBasedEstimator(self.s1)
        edges = se1.estimate_structure(
                                    max_parents = None,
                                    iterations_number = 100,
                                    patience = 35,
                                    tabu_length = 15,
                                    tabu_rules_duration = 15,
                                    optimizer = 'hill',
                                    disable_multiprocessing=False
                                    )
        

        self.assertEqual(edges, true_edges)

    def test_structure_2(self):
        with open("./PyCTBN/test_data/networks_and_trajectories_binary_data_02_10_1.json") as f:
                    raw_data = json.load(f)

                    trajectory_list_raw= raw_data["samples"]

                    trajectory_list = [pd.DataFrame(sample) for sample in trajectory_list_raw]

                    variables= pd.DataFrame(raw_data["variables"])
                    prior_net_structure = pd.DataFrame(raw_data["dyn.str"])


        self.importer = SampleImporter(
                                        trajectory_list=trajectory_list,
                                        variables=variables,
                                        prior_net_structure=prior_net_structure
                                    )
        
        self.importer.import_data()
        #cls.s1 = sp.SamplePath(cls.importer)

        #cls.traj = cls.s1.concatenated_samples

       # print(len(cls.traj))
        self.s1 = SamplePath(self.importer)
        self.s1.build_trajectories()
        self.s1.build_structure()

        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))
    
        se1 = StructureScoreBasedEstimator(self.s1)
        edges = se1.estimate_structure(
                                    max_parents = None,
                                    iterations_number = 100,
                                    patience = 35,
                                    tabu_length = 15,
                                    tabu_rules_duration = 15,
                                    optimizer = 'hill',
                                    disable_multiprocessing=True
                                    )
        
        'calculate precision and recall'
        n_missing_edges = 0
        n_added_fake_edges = 0

       
        n_added_fake_edges = len(edges.difference(true_edges))

        n_missing_edges = len(true_edges.difference(edges))

        n_true_positive = len(true_edges) - n_missing_edges

        precision = n_true_positive / (n_true_positive + n_added_fake_edges)

        recall = n_true_positive / (n_true_positive + n_missing_edges)

        self.assertGreaterEqual(precision,0.75)
        self.assertGreaterEqual(recall,0.75)

    def test_structure_3(self):
        with open("./PyCTBN/test_data/networks_and_trajectories_ternary_data_01_6_1.json") as f:
                    raw_data = json.load(f)

                    trajectory_list_raw= raw_data["samples"]

                    trajectory_list = [pd.DataFrame(sample) for sample in trajectory_list_raw]

                    variables= raw_data["variables"]
                    prior_net_structure = raw_data["dyn.str"]


        self.importer = SampleImporter(
                                        trajectory_list=trajectory_list,
                                        variables=variables,
                                        prior_net_structure=prior_net_structure
                                    )
        
        self.importer.import_data()
        #cls.s1 = sp.SamplePath(cls.importer)

        #cls.traj = cls.s1.concatenated_samples

       # print(len(cls.traj))
        self.s1 = SamplePath(self.importer)
        self.s1.build_trajectories()
        self.s1.build_structure()

        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))

        known_edges = self.s1.structure.edges[0:2]

        se1 = StructureScoreBasedEstimator(self.s1,known_edges=known_edges)
        edges = se1.estimate_structure(
                                    max_parents = 4,
                                    iterations_number = 100,
                                    patience = 35,
                                    tabu_length = 15,
                                    tabu_rules_duration = 15,
                                    optimizer = 'hill',
                                    disable_multiprocessing=True
                                    )
        
        'calculate precision and recall'
        n_missing_edges = 0
        n_added_fake_edges = 0

       
        n_added_fake_edges = len(edges.difference(true_edges))

        n_missing_edges = len(true_edges.difference(edges))

        n_true_positive = len(true_edges) - n_missing_edges

        precision = n_true_positive / (n_true_positive + n_added_fake_edges)

        recall = n_true_positive / (n_true_positive + n_missing_edges)

        self.assertGreaterEqual(precision,0.75)
        self.assertGreaterEqual(recall,0.75)      


if __name__ == '__main__':
    unittest.main()

