
import glob
import math
import os
import unittest

import networkx as nx
import numpy as np
import psutil
from line_profiler import LineProfiler
import copy


from ...classes.structure_graph.sample_path import SamplePath
from ...classes.estimators.structure_score_based_estimator import StructureScoreBasedEstimator
from ...classes.utility.json_importer import JsonImporter



class TestHillClimbingSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #cls.read_files = glob.glob(os.path.join('../../data', "*.json"))


        cls.importer = JsonImporter("./main_package/data/networks_and_trajectories_binary_data_01_10.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.importer.import_data(0)
        cls.s1 = SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()



    def test_structure(self):
        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))

        se1 = StructureScoreBasedEstimator(self.s1)
        edges = se1.estimate_structure(
                            max_parents = None,
                            iterations_number = 40,
                            patience = None,
                            optimizer = 'hill'
                            )
        

        self.assertEqual(edges, true_edges)
        


if __name__ == '__main__':
    unittest.main()

