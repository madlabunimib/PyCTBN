import unittest
import numpy as np
import itertools

import set_of_cims as soci


class TestSetOfCims(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.node_id = 'X'
        cls.possible_cardinalities = [2, 3]
        #cls.possible_states = [[0,1], [0, 1, 2]]
        cls.node_states_number = range(2, 4)

    def test_init(self):
        # empty parent set
        for sn in self.node_states_number:
            self.aux_test_init(self.node_id, [], sn)
        # one parent
        for sn in self.node_states_number:
            for p in itertools.product(self.possible_cardinalities, repeat=1):
                self.aux_test_init(self.node_id, list(p), sn)
        #two parents
        for sn in self.node_states_number:
            for p in itertools.product(self.possible_cardinalities, repeat=2):
                self.aux_test_init(self.node_id, list(p), sn)

    def test_indexes_converter(self):
        # empty parent set
        for sn in self.node_states_number:
            self.aux_test_indexes_converter(self.node_id, [], sn)
        # one parent
        for sn in self.node_states_number:
            for p in itertools.product(self.possible_cardinalities, repeat=1):
                self.aux_test_init(self.node_id, list(p), sn)
        # two parents
        for sn in self.node_states_number:
            for p in itertools.product(self.possible_cardinalities, repeat=2):
                self.aux_test_init(self.node_id, list(p), sn)

    def aux_test_indexes_converter(self, node_id, parents_states_number, node_states_number):
        sofcims = soci.SetOfCims(node_id, parents_states_number, node_states_number)
        if not parents_states_number:
            self.assertEqual(sofcims.indexes_converter([]), 0)
        else:
            parents_possible_values = []
            for cardi in parents_states_number:
                parents_possible_values.extend(range(0, cardi))
            for p in itertools.permutations(parents_possible_values, len(parents_states_number)):
                self.assertEqual(sofcims.indexes_converter(list(p)), np.ravel_multi_index(list(p), parents_states_number))

    def test_build_cims(self):
        state_res_times_list = []
        transition_matrices_list = []
        so1 = soci.SetOfCims('X',[3], 3)
        for i in range(0, 3):
            state_res_times = np.random.rand(1, 3)[0]
            state_res_times = state_res_times * 1000
            state_transition_matrix = np.random.randint(1, 10000, (3, 3))
            state_res_times_list.append(state_res_times)
            transition_matrices_list.append(state_transition_matrix)
        so1.build_cims(state_res_times_list, transition_matrices_list)
        self.assertEqual(len(state_res_times_list), so1.get_cims_number())
        self.assertIsNone(so1.transition_matrices)
        self.assertIsNone(so1.state_residence_times)

    def aux_test_init(self, node_id, parents_states_number, node_states_number):
        sofcims = soci.SetOfCims(node_id, parents_states_number, node_states_number)
        self.assertEqual(sofcims.node_id, node_id)
        self.assertTrue(np.array_equal(sofcims.parents_states_number, parents_states_number))
        self.assertEqual(sofcims.node_states_number, node_states_number)
        self.assertFalse(sofcims.actual_cims)
        self.assertEqual(sofcims.state_residence_times.shape[0], np.prod(np.array(parents_states_number)))
        self.assertEqual(len(sofcims.state_residence_times[0]),node_states_number)
        self.assertEqual(sofcims.transition_matrices.shape[0], np.prod(np.array(parents_states_number)))
        self.assertEqual(len(sofcims.transition_matrices[0][0]), node_states_number)



if __name__ == '__main__':
    unittest.main()
