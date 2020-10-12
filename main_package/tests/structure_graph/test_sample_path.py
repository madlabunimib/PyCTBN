import sys
sys.path.append("../../classes/")
import unittest
import glob
import os
import utility.json_importer as ji
import structure_graph.sample_path as sp
import structure_graph.trajectory as tr
import structure_graph.structure as st


class TestSamplePath(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.read_files = glob.glob(os.path.join('../../data', "*.json"))
        cls.importer = ji.JsonImporter(cls.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')

    def test_init(self):
        s1 = sp.SamplePath(self.importer)
        s1.build_trajectories()
        self.assertIsNotNone(s1.trajectories)
        self.assertIsInstance(s1.trajectories, tr.Trajectory)
        s1.build_structure()
        self.assertIsNotNone(s1.structure)
        self.assertIsInstance(s1.structure, st.Structure)
        self.assertTrue(s1.importer.concatenated_samples.empty)
        self.assertEqual(s1.total_variables_count, len(s1.importer.sorter))
        print(s1.structure)
        print(s1.trajectories)


if __name__ == '__main__':
    unittest.main()
