import unittest

import sample_path as sp
import structure_estimator as se

class TestStructureEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.s1 = sp.SamplePath('../data', 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.s1.build_trajectories()
        cls.s1.build_structure()

    def test_init(self):
        se1 = se.StructureEstimator(self.s1)
        self.assertEqual(self.s1, se1.sample_path)
        self.assertEqual(se1.complete_graph_frame.shape[0],
                         self.s1.total_variables_count *(self.s1.total_variables_count - 1))

    def test_one_iteration(self):
        se1 = se.StructureEstimator(self.s1, 0.1, 0.1)
        #se1.one_iteration_of_CTPC_algorithm('X')
        #self.aux_test_complete_test(se1, 'X', 'Y', ['Z'])
        se1.ctpc_algorithm()
        print(se1.complete_graph.edges)

    def aux_test_complete_test(self, estimator, test_par, test_child, p_set):
        estimator.complete_test(test_par, test_child, p_set)


if __name__ == '__main__':
    unittest.main()
