
import glob
import math
import os
import unittest

import networkx as nx
import numpy as np
import psutil
from line_profiler import LineProfiler

from ...PyCTBN.utility.cache import Cache
from ...PyCTBN.structure_graph.sample_path import SamplePath
from ...PyCTBN.estimators.structure_constraint_based_estimator import StructureConstraintBasedEstimator
from ...PyCTBN.utility.json_importer import JsonImporter

from multiprocessing import set_start_method

import copy


class TestStructureConstraintBasedEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_structure(self):
        #cls.read_files = glob.glob(os.path.join('../../data', "*.json"))
        self.importer = JsonImporter("./PyCTBN/test_data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        self.s1 = SamplePath(self.importer)
        self.s1.build_trajectories()
        self.s1.build_structure()

        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))

        
        se1 = StructureConstraintBasedEstimator(self.s1,0.1,0.1)
        edges = se1.estimate_structure(disable_multiprocessing=False)       


        self.importer = JsonImporter("./PyCTBN/test_data/networks_and_trajectories_binary_data_01_3.json", 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        self.s1 = SamplePath(self.importer)
        self.s1.build_trajectories()
        self.s1.build_structure()

        true_edges = copy.deepcopy(self.s1.structure.edges)
        true_edges = set(map(tuple, true_edges))

        
        se1 = StructureConstraintBasedEstimator(self.s1,0.1,0.1)
        edges = se1.estimate_structure(disable_multiprocessing=True)  
        
        

        self.assertEqual(edges, true_edges)

if __name__ == '__main__':
    unittest.main()
