
import unittest
import glob
import os
import random

from ..PyCTBN.json_importer import JsonImporter
from ..PyCTBN.sample_path import SamplePath
from ..PyCTBN.trajectory import Trajectory
from ..PyCTBN.structure import Structure


class TestSamplePath(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.read_files = glob.glob(os.path.join('./data', "*.json"))

    def test_init_not_initialized_importer(self):
        importer = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        self.assertRaises(RuntimeError, SamplePath, importer)

    def test_init_not_filled_dataframse(self):
        importer = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        importer.import_data(0)
        importer.clear_concatenated_frame()
        self.assertRaises(RuntimeError, SamplePath, importer)

    def test_init(self):
        importer = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        importer.import_data(0)
        s1 = SamplePath(importer)
        self.assertIsNone(s1.trajectories)
        self.assertIsNone(s1.structure)
        self.assertFalse(s1._importer.concatenated_samples.empty)
        self.assertIsNone(s1._total_variables_count)

    def test_build_trajectories(self):
        importer = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        importer.import_data(0)
        s1 = SamplePath(importer)
        s1.build_trajectories()
        self.assertIsInstance(s1.trajectories, Trajectory)

    def test_build_structure(self):
        importer = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        importer.import_data(0)
        s1 = SamplePath(importer)
        s1.build_structure()
        self.assertIsInstance(s1.structure, Structure)
        self.assertEqual(s1._total_variables_count, len(s1._importer.sorter))

    def test_build_structure_bad_sorter(self):
        importer = JsonImporter(self.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        importer.import_data(0)
        s1 = SamplePath(importer)
        random.shuffle(importer._sorter)
        self.assertRaises(RuntimeError, s1.build_structure)


if __name__ == '__main__':
    unittest.main()
