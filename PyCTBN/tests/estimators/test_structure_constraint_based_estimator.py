
# License: MIT License


import glob
import math
import os
import unittest

import networkx as nx
import numpy as np
import psutil

import json
import pandas as pd


from ...PyCTBN.structure_graph.sample_path import SamplePath
from ...PyCTBN.estimators.structure_constraint_based_estimator import StructureConstraintBasedEstimator
from ...PyCTBN.utility.sample_importer import SampleImporter

import copy


class TestStructureConstraintBasedEstimator(unittest.TestCase):
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

    def test_structure_1(self):
        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))

        se1 = StructureConstraintBasedEstimator(self.s1,0.1,0.1)
        edges = se1.estimate_structure(processes_number=2)
        
        self.assertFalse(se1.spurious_edges())
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

        se1 = StructureConstraintBasedEstimator(self.s1,0.1,0.1)
        edges = se1.estimate_structure(True)
        

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

        se1 = StructureConstraintBasedEstimator(self.s1,0.1,0.1)
        edges = se1.estimate_structure(True)
        

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
