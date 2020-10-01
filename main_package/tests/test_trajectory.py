import sys
sys.path.append("../classes/")
import unittest
import numpy as np

import trajectory as tr


class TestTrajectory(unittest.TestCase):

    def test_init(self):
        cols_list = [np.array([1.2,1.3,.14]), np.arange(1,4), np.arange(4,7)]
        t1 = tr.Trajectory(cols_list, len(cols_list) - 2)
        self.assertTrue(np.array_equal(cols_list[0], t1.times))
        self.assertTrue(np.array_equal(np.ravel(t1.complete_trajectory[:, : 1]), cols_list[1]))
        self.assertTrue(np.array_equal(np.ravel(t1.complete_trajectory[:, 1: 2]), cols_list[2]))
        self.assertEqual(len(cols_list) - 1, t1.complete_trajectory.shape[1])
        self.assertEqual(t1.size(), t1.times.size)

    def test_init_first_array_not_float_type(self):
        cols_list = [np.arange(1, 4), np.arange(4, 7), np.array([1.2, 1.3, .14])]
        self.assertRaises(TypeError, tr.Trajectory, cols_list, len(cols_list))

    def test_complete_trajectory(self):
        cols_list = [np.array([1.2, 1.3, .14]), np.arange(1, 4), np.arange(4, 7)]
        t1 = tr.Trajectory(cols_list, len(cols_list) - 2)
        complete = np.column_stack((cols_list[1], cols_list[2]))
        self.assertTrue(np.array_equal(t1.complete_trajectory, complete))

    def test_trajectory(self):
        cols_list = [np.array([1.2, 1.3, .14]), np.arange(1, 4), np.arange(4, 7)]
        t1 = tr.Trajectory(cols_list, len(cols_list) - 2)
        self.assertTrue(np.array_equal(cols_list[1], t1.trajectory.ravel()))

    def test_times(self):
        cols_list = [np.array([1.2, 1.3, .14]), np.arange(1, 4), np.arange(4, 7)]
        t1 = tr.Trajectory(cols_list, len(cols_list) - 2)
        self.assertTrue(np.array_equal(cols_list[0], t1.times))

    def test_repr(self):
        cols_list = [np.array([1.2, 1.3, .14]), np.arange(1, 4), np.arange(4, 7)]
        t1 = tr.Trajectory(cols_list, len(cols_list) - 2)
        print(t1)


if __name__ == '__main__':
    unittest.main()
