
import unittest
import numpy as np
from ...PyCTBN.structure_graph.structure import Structure


class TestStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = ['X','Y','Z']
        cls.indxs = np.array([0,1,2])
        cls.vals = np.array([3,3,3])
        cls.edges = [('X','Z'),('Y','Z'), ('Z','Y')]
        cls.vars_numb = len(cls.labels)

    def test_init(self):
        s1 = Structure(self.labels, self.indxs, self.vals, self.edges, self.vars_numb)
        self.assertListEqual(self.labels,s1.nodes_labels)
        self.assertIsInstance(s1.nodes_indexes, np.ndarray)
        self.assertTrue(np.array_equal(self.indxs, s1.nodes_indexes))
        self.assertIsInstance(s1.nodes_values, np.ndarray)
        self.assertTrue(np.array_equal(self.vals, s1.nodes_values))
        self.assertListEqual(self.edges, s1.edges)
        self.assertEqual(self.vars_numb, s1.total_variables_number)

    def test_get_node_id(self):
        s1 = Structure(self.labels, self.indxs, self.vals, self.edges, self.vars_numb)
        for indx, var in enumerate(self.labels):
            self.assertEqual(var, s1.get_node_id(indx))

    def test_edges_operations(self):
        s1 = Structure(self.labels, self.indxs, self.vals, self.edges, self.vars_numb)

        self.assertTrue(s1.contains_edge(('X','Z')))
        s1.add_edge(('Z','X'))
        self.assertTrue(s1.contains_edge(('Z','X')))
        s1.remove_edge(('Z','X'))
        self.assertFalse(s1.contains_edge(('Z','X')))

    def test_get_node_indx(self):
        l2 = self.labels[:]
        l2.remove('Y')
        i2 = self.indxs.copy()
        np.delete(i2, 1)
        v2 = self.vals.copy()
        np.delete(v2, 1)
        e2 = [('X','Z')]
        n2 = self.vars_numb - 1
        s1 = Structure(l2, i2, v2, e2, n2)
        for indx, var in zip(i2, l2):
            self.assertEqual(indx, s1.get_node_indx(var))

    def test_get_positional_node_indx(self):
        l2 = self.labels[:]
        l2.remove('Y')
        i2 = self.indxs.copy()
        np.delete(i2, 1)
        v2 = self.vals.copy()
        np.delete(v2, 1)
        e2 = [('X', 'Z')]
        n2 = self.vars_numb - 1
        s1 = Structure(l2, i2, v2, e2, n2)
        for indx, var in enumerate(s1.nodes_labels):
            self.assertEqual(indx, s1.get_positional_node_indx(var))

    def test_get_states_number(self):
        l2 = self.labels[:]
        l2.remove('Y')
        i2 = self.indxs.copy()
        np.delete(i2, 1)
        v2 = self.vals.copy()
        np.delete(v2, 1)
        e2 = [('X', 'Z')]
        n2 = self.vars_numb - 1
        s1 = Structure(l2, i2, v2, e2, n2)
        for val, node in zip(v2, l2):
            self.assertEqual(val, s1.get_states_number(node))

    def test_equality(self):
        s1 = Structure(self.labels, self.indxs, self.vals, self.edges, self.vars_numb)
        s2 = Structure(self.labels, self.indxs, self.vals, self.edges, self.vars_numb)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1,4)

    def test_repr(self):
        s1 = Structure(self.labels, self.indxs, self.vals, self.edges, self.vars_numb)
        print(s1)


if __name__ == '__main__':
    unittest.main()
