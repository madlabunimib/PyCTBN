
import unittest
import numpy as np

from ...classes.structure_graph.conditional_intensity_matrix import ConditionalIntensityMatrix


class TestConditionalIntensityMatrix(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.state_res_times = np.random.rand(1, 3)[0]
        cls.state_res_times = cls.state_res_times * 1000
        cls.state_transition_matrix = np.random.randint(1, 10000, (3, 3))
        for i in range(0, len(cls.state_res_times)):
            cls.state_transition_matrix[i, i] = 0
            cls.state_transition_matrix[i, i] = np.sum(cls.state_transition_matrix[i])

    def test_init(self):
        c1 = ConditionalIntensityMatrix(self.state_res_times, self.state_transition_matrix)
        self.assertTrue(np.array_equal(self.state_res_times, c1.state_residence_times))
        self.assertTrue(np.array_equal(self.state_transition_matrix, c1.state_transition_matrix))
        self.assertEqual(c1.cim.dtype, np.float)
        self.assertEqual(self.state_transition_matrix.shape, c1.cim.shape)

    def test_compute_cim_coefficients(self):
        c1 = ConditionalIntensityMatrix(self.state_res_times, self.state_transition_matrix)
        c2 = self.state_transition_matrix.astype(np.float)
        np.fill_diagonal(c2, c2.diagonal() * -1)
        for i in range(0, len(self.state_res_times)):
            for j in range(0, len(self.state_res_times)):
                c2[i, j] = (c2[i, j] + 1) / (self.state_res_times[i] + 1)
        c1.compute_cim_coefficients()
        for i in range(0, len(c1.state_residence_times)):
            self.assertTrue(np.isclose(np.sum(c1.cim[i]), 0.0, 1e-02, 1e-01))
        for i in range(0, len(self.state_res_times)):
            for j in range(0, len(self.state_res_times)):
                self.assertTrue(np.isclose(c1.cim[i, j], c2[i, j], 1e-02, 1e-01))

    def test_repr(self):
        c1 = ConditionalIntensityMatrix(self.state_res_times, self.state_transition_matrix)
        print(c1)


if __name__ == '__main__':
    unittest.main()
