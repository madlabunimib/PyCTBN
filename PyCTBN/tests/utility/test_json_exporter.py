import unittest
import random
import numpy as np
import os.path

from PyCTBN.PyCTBN.utility.json_exporter import JsonExporter
from PyCTBN.PyCTBN.structure_graph.trajectory_generator import TrajectoryGenerator
from PyCTBN.PyCTBN.structure_graph.network_generator import NetworkGenerator

class TestJSONExporter(unittest.TestCase):
    def test_generate_graph(self):
        ng = NetworkGenerator(["X", "Y", "Z"], [3 for i in range(3)])
        ng.generate_graph(0.3)
        ng.generate_cims(1, 3)
        e1 = JsonExporter(ng.variables, ng.dyn_str, ng.cims)
        tg = TrajectoryGenerator(variables = ng.variables, dyn_str = ng.dyn_str, dyn_cims = ng.cims)
        n_traj = random.randint(1, 30)
        for i in range(n_traj):
            sigma = tg.CTBN_Sample(max_tr = 100)
            e1.add_trajectory(sigma)

        self.assertEqual(n_traj, len(e1._trajectories))
        e1.out_file("test.json")
        
        self.assertTrue(os.path.isfile("test.json"))

if __name__ == '__main__':
    unittest.main()