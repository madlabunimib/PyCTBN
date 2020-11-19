import sys
sys.path.append("../classes/")
import unittest
import glob
import os
import json_importer as ji
import sample_path as sp
import trajectory as tr
import structure as st


class TestSamplePath(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.read_files = glob.glob(os.path.join('../data', "*.json"))
        cls.importer = ji.JsonImporter(cls.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')

    def test_init(self):
        s1 = sp.SamplePath(self.importer)
        self.assertIsNone(s1.trajectories)
        self.assertIsNone(s1.structure)
        self.assertFalse(s1.importer.concatenated_samples.empty)
        self.assertIsNone(s1.total_variables_count)

    def test_build_trajectories(self):
        s1 = sp.SamplePath(self.importer)
        s1.build_trajectories()
        self.assertIsInstance(s1.trajectories, tr.Trajectory)

    def test_build_structure(self):
        s1 = sp.SamplePath(self.importer)
        s1.build_structure()
        self.assertIsInstance(s1.structure, st.Structure)
        self.assertEqual(s1.total_variables_count, len(s1.importer.sorter))

if __name__ == '__main__':
    unittest.main()
