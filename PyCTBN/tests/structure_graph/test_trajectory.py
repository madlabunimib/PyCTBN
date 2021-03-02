
import unittest
import numpy as np
import glob
import os

from ...PyCTBN.structure_graph.trajectory import Trajectory
from ...PyCTBN.utility.json_importer import JsonImporter

class TestTrajectory(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.read_files = glob.glob(os.path.join('./PyCTBN/test_data', "*.json"))
        cls.importer = JsonImporter(cls.read_files[2], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.importer.import_data(0)

    def test_init(self):
        t1 = Trajectory(self.importer.build_list_of_samples_array(self.importer.concatenated_samples),
                len(self.importer.sorter) + 1)
        self.assertTrue(np.array_equal(self.importer.concatenated_samples.iloc[:, 0].to_numpy(), t1.times))
        self.assertTrue(np.array_equal(self.importer.concatenated_samples.iloc[:,1:].to_numpy(), t1.complete_trajectory))
        self.assertTrue(np.array_equal(self.importer.concatenated_samples.iloc[:, 1: len(self.importer.sorter) + 1], t1.trajectory))
        self.assertEqual(len(self.importer.sorter) + 1, t1._original_cols_number)

        self.assertEqual(self.importer.concatenated_samples.iloc[:,1:].to_numpy().shape[0], t1.size())
        print(t1)

if __name__ == '__main__':
    unittest.main()
