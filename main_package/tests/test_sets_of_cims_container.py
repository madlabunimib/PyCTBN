import unittest

import sets_of_cims_container as scc
import set_of_cims as sc


class TestSetsOfCimsContainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.variables = ['X', 'Y', 'Z']
        cls.states_per_node = [3, 3, 3]
        cls.parents_states_list = [[], [3], [3, 3]]

    def test_init(self):
        c1 = scc.SetsOfCimsContainer(self.variables, self.states_per_node, self.parents_states_list)
        self.assertEqual(len(c1.sets_of_cims), len(self.variables))
        for set_of_cims in c1.sets_of_cims:
            self.assertIsInstance(set_of_cims, sc.SetOfCims)



if __name__ == '__main__':
    unittest.main()
