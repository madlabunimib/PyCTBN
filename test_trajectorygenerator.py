import unittest
import random

from PyCTBN.PyCTBN.structure_graph.trajectory_generator import TrajectoryGenerator
from PyCTBN.PyCTBN.utility.json_importer import JsonImporter

class TestTrajectoryGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.j1 = JsonImporter("./PyCTBN/test_data/networks_and_trajectories_binary_data_01_3.json", "samples", "dyn.str", "variables", "Time", "Name")

    def test_init(self):
        tg = TrajectoryGenerator(self.j1)
        self.assertEqual(len(tg._vnames), len(self.j1.variables))
        self.assertIsInstance(tg._vnames, list)
        self.assertIsInstance(tg._parents, dict)
        self.assertIsInstance(tg._cims, dict)
        self.assertListEqual(list(tg._parents.keys()), tg._vnames)
        self.assertListEqual(list(tg._cims.keys()), tg._vnames)

    def test_generated_trajectory(self):
        tg = TrajectoryGenerator(self.j1)
        end_time = random.randint(5, 100)
        sigma = tg.CTBN_Sample(end_time)
        self.assertLessEqual(sigma.times.loc[len(sigma) - 1].at["Time"], end_time)
        for index, row in sigma.iterrows():
            if index > 0:
                self.assertLess(sigma.times.loc[index - 1].at["Time"], row.at["Time"])
                diff = abs(sum(sigma.loc[index - 1, sigma.columns != "Time"]) - 
                    sum(sigma.loc[index, sigma.columns != "Time"]))
                self.assertEqual(diff, 1)

unittest.main()