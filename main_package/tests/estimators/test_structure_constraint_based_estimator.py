
import glob
import math
import os
import unittest

import networkx as nx
import numpy as np
import psutil
from line_profiler import LineProfiler

import json
import pandas as pd


from ...classes.structure_graph.sample_path import SamplePath
from ...classes.estimators.structure_constraint_based_estimator import StructureConstraintBasedEstimator
from ...classes.utility.sample_importer import SampleImporter

import copy


class TestStructureConstraintBasedEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("./main_package/data/networks_and_trajectories_ternary_data_3.json") as f:
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

    def test_structure(self):
        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))

        se1 = StructureConstraintBasedEstimator(self.s1,0.1,0.1)
        edges = se1.estimate_structure(disable_multiprocessing=False)
        

        self.assertEqual(edges, true_edges)

if __name__ == '__main__':
    unittest.main()
