import unittest
import random

from PyCTBN.PyCTBN.structure_graph.trajectory import Trajectory
from PyCTBN.PyCTBN.structure_graph.trajectory_generator import TrajectoryGenerator
from PyCTBN.PyCTBN.utility.json_importer import JsonImporter

class TestTrajectoryGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.j1 = JsonImporter(file_path = "./PyCTBN/test_data/networks_and_trajectories_binary_data_01_3.json", samples_label = "samples",
                            structure_label = "dyn.str", variables_label = "variables",
                            cims_label = "dyn.cims", time_key = "Time", 
                            variables_key = "Name")
        cls.j1.import_data(0)

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
        traj = Trajectory(self.j1.build_list_of_samples_array(sigma), len(self.j1.sorter) + 1)
        self.assertLessEqual(traj.times[len(traj.times) - 1], end_time)
        for index in range(len(traj.times)):
            if index > 0:
                self.assertLess(traj.times[index - 1], traj.times[index])
                if index < len(traj.times) - 1:
                    diff = abs(sum(traj.trajectory[index - 1]) - sum(traj.trajectory[index]))
                    self.assertEqual(diff, 1)
        self.assertEqual(sum(traj.trajectory[len(traj.times) - 1]), -1 * len(self.j1.sorter))

    def test_generated_trajectory_max_tr(self):
        tg = TrajectoryGenerator(self.j1)
        n_tr = random.randint(5, 100)
        sigma = tg.CTBN_Sample(max_tr = n_tr)
        traj = Trajectory(self.j1.build_list_of_samples_array(sigma), len(self.j1.sorter) + 1)
        self.assertEqual(len(traj.times), n_tr + 1)

    def test_multi_trajectory(self):
        tg = TrajectoryGenerator(self.j1)
        max_trs = [random.randint(5, 100) for i in range(10)]
        trajectories = tg.multi_trajectory(max_trs = max_trs)
        self.assertEqual(len(trajectories), len(max_trs))
        self.assertTrue({len(trajectory) for trajectory in trajectories} == {max_tr + 1 for max_tr in max_trs})
        t_ends = [random.randint(100, 500) for i in range(10)]
        trajectories = tg.multi_trajectory(t_ends = t_ends)
        self.assertEqual(len(trajectories), len(t_ends))

if __name__ == '__main__':
    unittest.main()