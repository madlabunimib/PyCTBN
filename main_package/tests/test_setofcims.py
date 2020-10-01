import sys
sys.path.append("../classes/")
import unittest
import numpy as np
import itertools

import set_of_cims as soci


class TestSetOfCims(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.node_id = 'X'
        cls.possible_cardinalities = [2, 3]
        cls.possible_states = [[0,1], [0, 1, 2]]
        cls.node_states_number = range(2, 4)

    def test_init(self):
        # empty parent set
        for sn in self.node_states_number:
            p_combs = self.build_p_comb_structure_for_a_node([])
            self.aux_test_init(self.node_id, [], sn, p_combs)
        # one parent
        for sn in self.node_states_number:
            for p in itertools.product(self.possible_cardinalities, repeat=1):
                p_combs = self.build_p_comb_structure_for_a_node(list(p))
                self.aux_test_init(self.node_id, list(p), sn, p_combs)
        #two parents
        for sn in self.node_states_number:
            for p in itertools.product(self.possible_cardinalities, repeat=2):
                p_combs = self.build_p_comb_structure_for_a_node(list(p))
                self.aux_test_init(self.node_id, list(p), sn, p_combs)

    def test_build_cims(self):
        # empty parent set
        for sn in self.node_states_number:
            p_combs = self.build_p_comb_structure_for_a_node([])
            self.aux_test_build_cims(self.node_id, [], sn, p_combs)
        # one parent
        for sn in self.node_states_number:
            for p in itertools.product(self.possible_cardinalities, repeat=1):
                p_combs = self.build_p_comb_structure_for_a_node(list(p))
                self.aux_test_build_cims(self.node_id, list(p), sn, p_combs)
        #two parents
        for sn in self.node_states_number:
            for p in itertools.product(self.possible_cardinalities, repeat=2):
                p_combs = self.build_p_comb_structure_for_a_node(list(p))
                self.aux_test_build_cims(self.node_id, list(p), sn, p_combs)

    def test_filter_cims_with_mask(self):
        p_combs = self.build_p_comb_structure_for_a_node(self.possible_cardinalities)
        sofc1 = soci.SetOfCims('X', self.possible_cardinalities, 3, p_combs)
        state_res_times_list = []
        transition_matrices_list = []
        for i in range(len(p_combs)):
            state_res_times = np.random.rand(1, 3)[0]
            state_res_times = state_res_times * 1000
            state_transition_matrix = np.random.randint(1, 10000, (3, 3))
            state_res_times_list.append(state_res_times)
            transition_matrices_list.append(state_transition_matrix)
        sofc1.build_cims(state_res_times_list, transition_matrices_list)
        for length_of_mask in range(3):
            for mask in list(itertools.permutations([True, False],r=length_of_mask)):
                m = np.array(mask)
                for parent_value in range(self.possible_cardinalities[0]):
                    cims = sofc1.filter_cims_with_mask(m, [parent_value])
                    if length_of_mask == 0 or length_of_mask == 1:
                        self.assertTrue(np.array_equal(sofc1.actual_cims, cims))
                    else:
                        indxs = self.another_filtering_method(p_combs, m, [parent_value])
                        self.assertTrue(np.array_equal(cims, sofc1.actual_cims[indxs]))

    def aux_test_build_cims(self, node_id, p_values, node_states, p_combs):
        state_res_times_list = []
        transition_matrices_list = []
        so1 = soci.SetOfCims(node_id, p_values, node_states, p_combs)
        for i in range(len(p_combs)):
            state_res_times = np.random.rand(1, node_states)[0]
            state_res_times = state_res_times * 1000
            state_transition_matrix = np.random.randint(1, 10000, (node_states, node_states))
            state_res_times_list.append(state_res_times)
            transition_matrices_list.append(state_transition_matrix)
        so1.build_cims(state_res_times_list, transition_matrices_list)
        self.assertEqual(len(state_res_times_list), so1.get_cims_number())
        self.assertIsInstance(so1.actual_cims, np.ndarray)
        self.assertIsNone(so1.transition_matrices)
        self.assertIsNone(so1.state_residence_times)

    def aux_test_init(self, node_id, parents_states_number, node_states_number, p_combs):
        sofcims = soci.SetOfCims(node_id, parents_states_number, node_states_number, p_combs)
        self.assertEqual(sofcims.node_id, node_id)
        self.assertTrue(np.array_equal(sofcims.p_combs, p_combs))
        self.assertTrue(np.array_equal(sofcims.parents_states_number, parents_states_number))
        self.assertEqual(sofcims.node_states_number, node_states_number)
        self.assertFalse(sofcims.actual_cims)
        self.assertEqual(sofcims.state_residence_times.shape[0], np.prod(np.array(parents_states_number)))
        self.assertEqual(len(sofcims.state_residence_times[0]),node_states_number)
        self.assertEqual(sofcims.transition_matrices.shape[0], np.prod(np.array(parents_states_number)))
        self.assertEqual(len(sofcims.transition_matrices[0][0]), node_states_number)

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

    def build_p_comb_structure_for_a_node(self, parents_values):
        """
        Builds the combinatory structure that contains the combinations of all the values contained in parents_values.

        Parameters:
            parents_values: the cardinalities of the nodes
        Returns:
            a numpy matrix containing a grid of the combinations
        """
        tmp = []
        for val in parents_values:
            tmp.append([x for x in range(val)])
        if len(parents_values) > 0:
            parents_comb = np.array(np.meshgrid(*tmp)).T.reshape(-1, len(parents_values))
            if len(parents_values) > 1:
                tmp_comb = parents_comb[:, 1].copy()
                parents_comb[:, 1] = parents_comb[:, 0].copy()
                parents_comb[:, 0] = tmp_comb
        else:
            parents_comb = np.array([[]], dtype=np.int)
        return parents_comb

    def another_filtering_method(self,p_combs, mask, parent_value):
        masked_combs = p_combs[:, mask]
        indxs = []
        for indx, val in enumerate(masked_combs):
            if val == parent_value:
                indxs.append(indx)
        return np.array(indxs)

if __name__ == '__main__':
    unittest.main()
