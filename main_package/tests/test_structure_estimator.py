import unittest
from line_profiler import LineProfiler
from multiprocessing import  Pool

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
        lp = LineProfiler()
        lp.add_function(se1.complete_test)
        lp.add_function(se1.one_iteration_of_CTPC_algorithm)
        lp.add_function(se1.independence_test)
        lp_wrapper = lp(se1.ctpc_algorithm)
        lp_wrapper()
        lp.print_stats()
        #se1.ctpc_algorithm()
        print(se1.complete_graph.edges)
        print(self.s1.structure.list_of_edges())

    def aux_test_complete_test(self, estimator, test_par, test_child, p_set):
        estimator.complete_test(test_par, test_child, p_set)


if __name__ == '__main__':
    unittest.main()
