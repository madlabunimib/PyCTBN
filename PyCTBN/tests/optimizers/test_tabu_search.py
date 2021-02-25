import sys
sys.path.append("../../PyCTBN/")
import glob
import math
import os
import unittest

import networkx as nx
import numpy as np
import pandas as pd
import psutil
from line_profiler import LineProfiler
import copy
import json

import utility.cache as ch
import structure_graph.sample_path as sp
import estimators.structure_score_based_estimator as se
import utility.json_importer as ji
import utility.sample_importer as si





class TestTabuSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #cls.read_files = glob.glob(os.path.join('../../data', "*.json"))

        with open("./PyCTBN/test_data/networks_and_trajectories_binary_data_01_3.json") as f:
            raw_data = json.load(f)

            trajectory_list_raw= raw_data[0]["samples"]

            trajectory_list = [pd.DataFrame(sample) for sample in trajectory_list_raw]

            variables= pd.DataFrame(raw_data[0]["variables"])
            prior_net_structure = pd.DataFrame(raw_data[0]["dyn.str"])


        cls.importer = si.SampleImporter(
                                        trajectory_list=trajectory_list,
                                        variables=variables,
                                        prior_net_structure=prior_net_structure
                                    )
        
        cls.importer.import_data()
        #cls.s1 = sp.SamplePath(cls.importer)

        #cls.traj = cls.s1.concatenated_samples

       # print(len(cls.traj))
        cls.s1 = sp.SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()
        #cls.s1.clear_memory() 



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

