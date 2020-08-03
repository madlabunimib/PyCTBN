import sys
sys.path.append("/Users/Zalum/Desktop/Tesi/CTBN_Project/main_package/classes/")
import unittest
import pandas as pd
import numpy as np

import sample_path as sp
import structure as st
import network_graph as ng
import parameters_estimator as pe


class TestStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = ['X','Y','Z']
        cls.indxs = np.array([0,1,2])
        cls.vals = np.array([3,3,3])
        cls.edges = [('X','Z'),('Y','Z'), ('Z','Y')]
        cls.vars_numb = len(cls.labels)

    def test_init(self):
        s1 = st.Structure(self.labels, self.indxs, self.vals, self.edges, self.vars_numb)
        self.assertListEqual(self.labels,s1.nodes_labels)
        self.assertTrue(np.array_equal(self.indxs, s1.nodes_indexes))
        self.assertTrue(np.array_equal(self.vals, s1.nodes_values))
        self.assertListEqual(self.edges, s1.edges)
        self.assertEqual(self.vars_numb, s1.total_variables_number)

    def test_get_node_id(self):
        s1 = st.Structure(self.labels, self.indxs, self.vals, self.edges, self.vars_numb)
        for indx, var in enumerate(self.labels):
            self.assertEqual(var, s1.get_node_id(indx))

    def test_get_node_indx(self):
        l2 = self.labels[:]
        l2.remove('Y')
        i2 = self.indxs.copy()
        np.delete(i2, 1)
        v2 = self.vals.copy()
        np.delete(v2, 1)
        e2 = [('X','Z')]
        n2 = self.vars_numb - 1
        s1 = st.Structure(l2, i2, v2, e2, n2)
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
        s1 = st.Structure(l2, i2, v2, e2, n2)
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
        s1 = st.Structure(l2, i2, v2, e2, n2)
        for val, node in zip(v2, l2):
            self.assertEqual(val, s1.get_states_number(node))
#TODO FORSE QUESTO TEST NON serve verificare se questo metodo sia davvero utile
    """def test_get_states_numeber_by_indx(self):
        s1 = st.Structure(self.structure_frame, self.variables_frame, len(self.variables_frame.index))
        for indx, row in self.variables_frame.iterrows():
            self.assertEqual(row[1], s1.get_states_number_by_indx(indx))

    def test_new_init(self):
        #self.variables_frame.drop(self.variables_frame[(self.variables_frame['Name'] == 'Y')].index, inplace=True)
        labels = self.variables_frame['Name'].to_list()
        indxs = self.variables_frame.index.to_numpy()
        vals = self.variables_frame['Value'].to_numpy()
        edges = list(self.structure_frame.to_records(index=False))
        print(labels)
        print(indxs)
        print(vals)
        print(edges)
        s1 = st.Structure(labels, indxs, vals, edges, len(self.variables_frame.index))
        #print(s1.get_node_id(2))
        print(s1.get_node_indx('Z'))
        print(s1.get_positional_node_indx('Z'))
        print(s1.get_states_number('Z'))
        print(s1.get_states_number_by_indx(1))
        [CIM:
[[-4.82318981  1.18421625  3.63997346]
 [ 4.44726473 -9.20141291  4.755239  ]
 [ 2.93950444  4.36292948 -7.30152554]], CIM:
[[-6.0336893   1.69212904  4.34235011]
 [ 3.32692085 -5.03977237  1.7137923 ]
 [ 3.65519241  3.81402509 -7.46819716]], CIM:
[[-6.78778897  1.98559721  4.80306557]
 [ 1.23811008 -6.26366842  5.0265376 ]
 [ 3.02720526  4.0256821  -7.05222539]]]
 array([ 3,  9, 27])
 array([3, 9])
 array([1, 2])
 array([4, 1, 2])

        sp1 = sp.SamplePath('../data', 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        sp1.build_trajectories()
        sp1.build_structure()
        st1 = st.Structure(['X','Y','Z'], np.array([0,1,2]), np.array([3,3,3]), [('Z','X'),('Y', 'X')], sp1.total_variables_count)
        g1 = ng.NetworkGraph(st1)
        g1.init_graph()
        print("M Vector",g1.transition_scalar_indexing_structure)
        print("Time Vecotr",g1.time_scalar_indexing_strucure)
        print("Time Filter",g1.time_filtering)
        print("M Filter",g1.transition_filtering)
        print(g1.p_combs)
        print("AGG STR", g1.aggregated_info_about_nodes_parents)
        p1 = pe.ParametersEstimator(sp1,g1)
        p1.init_sets_cims_container()
        p1.compute_parameters_for_node('X')
        #print(p1.sets_of_cims_struct.get_cims_of_node(0,[1,0]))
        print(p1.sets_of_cims_struct.sets_of_cims[1].actual_cims)
        #print(p1.sets_of_cims_struct.sets_of_cims[2].get_cims_where_parents_except_last_are_in_state(np.array([0])))
        #print(p1.sets_of_cims_struct.sets_of_cims[0].p_combs)"""


if __name__ == '__main__':
    unittest.main()
