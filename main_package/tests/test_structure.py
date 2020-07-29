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
    def setUp(self):
        self.structure_frame = pd.DataFrame([{"From":"X","To":"Z"}, {"From":"Y","To":"Z"},
                                             {"From":"Z","To":"Y"} ])
        self.variables_frame = pd.DataFrame([{"Name":"X","Value":3},{"Name":"Y","Value":3},{"Name":"Z","Value":3}])

    def test_init(self):
        s1 = st.Structure(self.structure_frame, self.variables_frame, len(self.variables_frame.index))
        self.assertTrue(self.structure_frame.equals(s1.structure_frame))
        self.assertTrue(self.variables_frame.equals(s1.variables_frame))
        self.assertEqual(self.variables_frame.columns.values[0], s1.name_label)
        self.assertEqual(self.variables_frame.columns.values[1], s1.value_label)
        #print(len(self.variables_frame.index))
        self.assertEqual(len(self.variables_frame.index), s1.total_variables_number)

    def test_list_of_edges(self):
        s1 = st.Structure(self.structure_frame, self.variables_frame, len(self.variables_frame.index))
        records = self.structure_frame.to_records(index=False)
        result = list(records)
        for e1, e2 in zip(result, s1.list_of_edges()):
           self.assertEqual(e1, e2)

    def test_list_of_nodes_labels(self):
        s1 = st.Structure(self.structure_frame, self.variables_frame, len(self.variables_frame.index))
        self.assertEqual(list(self.variables_frame['Name']), s1.list_of_nodes_labels())

    def test_get_node_id(self):
        s1 = st.Structure(self.structure_frame, self.variables_frame, len(self.variables_frame.index))
        for indx, var in enumerate(list(self.variables_frame['Name'])):
            self.assertEqual(var, s1.get_node_id(indx))

    def test_get_node_indx(self):
        filtered_frame = self.variables_frame.drop(self.variables_frame[self.variables_frame['Name'] == 'Y'].index)
        #print(filtered_frame)
        s1 = st.Structure(self.structure_frame, filtered_frame, len(self.variables_frame.index))
        for indx, var in zip(filtered_frame.index, filtered_frame['Name']):
            self.assertEqual(indx, s1.get_node_indx(var))

    def test_list_of_node_indxs(self):
        filtered_frame = self.variables_frame.drop(self.variables_frame[self.variables_frame['Name'] == 'Y'].index)
        # print(filtered_frame)
        s1 = st.Structure(self.structure_frame, filtered_frame, len(self.variables_frame.index))

        for indx1, indx2 in zip(filtered_frame.index, s1.list_of_nodes_indexes()):
            self.assertEqual(indx1, indx2)

    def test_get_positional_node_indx(self):
        filtered_frame = self.variables_frame.drop(self.variables_frame[self.variables_frame['Name'] == 'Y'].index)
        # print(filtered_frame)
        s1 = st.Structure(self.structure_frame, filtered_frame, len(self.variables_frame.index))
        for indx, var in enumerate(s1.list_of_nodes_labels()):
            self.assertEqual(indx, s1.get_positional_node_indx(var))

    def test_get_states_number(self):
        s1 = st.Structure(self.structure_frame, self.variables_frame, len(self.variables_frame.index))
        for indx, row in self.variables_frame.iterrows():
            self.assertEqual(row[1], s1.get_states_number(row[0]))

    def test_get_states_numeber_by_indx(self):
        s1 = st.Structure(self.structure_frame, self.variables_frame, len(self.variables_frame.index))
        for indx, row in self.variables_frame.iterrows():
            self.assertEqual(row[1], s1.get_states_number_by_indx(indx))

    def test_new_init(self):
        #self.variables_frame.drop(self.variables_frame[(self.variables_frame['Name'] == 'Y')].index, inplace=True)
        """labels = self.variables_frame['Name'].to_list()
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
        """
        sp1 = sp.SamplePath('../data', 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        sp1.build_trajectories()
        sp1.build_structure()
        st1 = st.Structure(['X', 'Y', 'Z'], np.array([0,1,2]), np.array([3,3,3]), [('X', 'Y'), ('Z', 'Y')], sp1.total_variables_count)
        g1 = ng.NetworkGraph(st1)
        g1.init_graph()
        print(g1.transition_scalar_indexing_structure)
        print(g1.time_scalar_indexing_strucure)
        print(g1.time_filtering)
        print(g1.transition_filtering)
        p1 = pe.ParametersEstimator(sp1,g1)
        p1.init_sets_cims_container()
        p1.compute_parameters_for_node('Y')
        print(p1.sets_of_cims_struct.sets_of_cims[1].actual_cims)


if __name__ == '__main__':
    unittest.main()
