import unittest
import pandas as pd
import structure as st


class TestStructure(unittest.TestCase):
    def setUp(self):
        self.structure_frame = pd.DataFrame([{"From":"X","To":"Z"}, {"From":"X","To":"Y"},{"From":"Y","To":"X"},
                                             {"From":"Y","To":"Z"},{"From":"Z","To":"Y"}, {"From":"Z","To":"X"} ])
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

if __name__ == '__main__':
    unittest.main()
