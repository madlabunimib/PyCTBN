import unittest
import sample_path as sp
import trajectory as tr
import structure as st


class TestSamplePath(unittest.TestCase):

    def test_init(self):
        s1 = sp.SamplePath('../data', 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        s1.build_trajectories()
        self.assertIsNotNone(s1.trajectories)
        self.assertIsInstance(s1.trajectories, tr.Trajectory)
        s1.build_structure()
        self.assertIsNotNone(s1.structure)
        self.assertIsInstance(s1.structure, st.Structure)
        self.assertTrue(s1.importer.concatenated_samples.empty)
        print(s1.structure)
        print(s1.trajectories)




if __name__ == '__main__':
    unittest.main()
