
import unittest
import glob
import os

from ..PyCTBN.json_importer import JsonImporter
from ..PyCTBN.sample_path import SamplePath
from ..PyCTBN.trajectory import Trajectory
from ..PyCTBN.structure import Structure


class TestSamplePath(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.read_files = glob.glob(os.path.join('./data', "*.json"))
        cls.importer = JsonImporter(cls.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)

    def test_init(self):
        s1 = SamplePath(self.importer)
        self.assertIsNone(s1.trajectories)
        self.assertIsNone(s1.structure)
        self.assertFalse(s1._importer.concatenated_samples.empty)
        self.assertIsNone(s1._total_variables_count)

    def test_build_trajectories(self):
        s1 = SamplePath(self.importer)
        s1.build_trajectories()
        self.assertIsInstance(s1.trajectories, Trajectory)

    def test_build_structure(self):
        s1 = SamplePath(self.importer)
        s1.build_structure()
        self.assertIsInstance(s1.structure, Structure)
        self.assertEqual(s1._total_variables_count, len(s1._importer.sorter))


if __name__ == '__main__':
    unittest.main()
