import sys
sys.path.append("../classes/")
import unittest
import glob
import os
import networkx as nx
import numpy as np
import itertools

import sample_path as sp
import network_graph as ng
import json_importer as ji


class TestNetworkGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.read_files = glob.glob(os.path.join('../data', "*.json"))
        cls.importer = ji.JsonImporter(cls.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 0)
        cls.s1 = sp.SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()

    def test_init(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        self.assertEqual(self.s1.structure, g1._graph_struct)
        self.assertIsInstance(g1._graph, nx.DiGraph)
        self.assertTrue(np.array_equal(g1._nodes_indexes, self.s1.structure.nodes_indexes))
        self.assertListEqual(g1._nodes_labels, self.s1.structure.nodes_labels)
        self.assertTrue(np.array_equal(g1._nodes_values, self.s1.structure.nodes_values))
        self.assertIsNone(g1._fancy_indexing)
        self.assertIsNone(g1.time_scalar_indexing_strucure)
        self.assertIsNone(g1.transition_scalar_indexing_structure)
        self.assertIsNone(g1.transition_filtering)
        self.assertIsNone(g1.p_combs)

    def test_add_nodes(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        for n1, n2 in zip(g1.nodes, self.s1.structure.nodes_labels):
            self.assertEqual(n1, n2)

    def test_add_edges(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_edges(self.s1.structure.edges)
        for e in self.s1.structure.edges:
            self.assertIn(tuple(e), g1.edges)

    def aux_aggregated_par_list_data(self, graph, node_id, sorted_par_list_aggregated_info):
        for indx, element in enumerate(sorted_par_list_aggregated_info):
            if indx == 0:
                self.assertEqual(graph.get_parents_by_id(node_id), element)
                for j in range(0, len(sorted_par_list_aggregated_info[0]) - 1):
                    self.assertLess(self.s1.structure.get_node_indx(sorted_par_list_aggregated_info[0][j]),
                                    self.s1.structure.get_node_indx(sorted_par_list_aggregated_info[0][j + 1]))
            elif indx == 1:
                for node, node_indx in zip(sorted_par_list_aggregated_info[0], sorted_par_list_aggregated_info[1]):
                    self.assertEqual(graph.get_node_indx(node), node_indx)
            else:
                for node, node_val in zip(sorted_par_list_aggregated_info[0], sorted_par_list_aggregated_info[2]):
                    self.assertEqual(graph._graph_struct.get_states_number(node), node_val)

    def test_get_ord_set_of_par_of_all_nodes(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        sorted_list_of_par_lists = g1.get_ord_set_of_par_of_all_nodes()
        for node, par_list in zip(g1.nodes, sorted_list_of_par_lists):
            self.aux_aggregated_par_list_data(g1, node, par_list)


    def aux_build_time_scalar_indexing_structure_for_a_node(self, graph, node_id, parents_indxs, parents_labels, parents_vals):
        time_scalar_indexing = graph.build_time_scalar_indexing_structure_for_a_node(node_id, parents_vals)
        self.assertEqual(len(time_scalar_indexing), len(parents_indxs) + 1)
        merged_list = parents_labels[:]
        merged_list.insert(0, node_id)
        vals_list = []
        for node in merged_list:
            vals_list.append(graph.get_states_number(node))
        t_vec = np.array(vals_list)
        t_vec = t_vec.cumprod()
        self.assertTrue(np.array_equal(time_scalar_indexing, t_vec))

    def aux_build_transition_scalar_indexing_structure_for_a_node(self, graph, node_id, parents_indxs, parents_labels,
                                                                  parents_values):
        transition_scalar_indexing = graph.build_transition_scalar_indexing_structure_for_a_node(node_id,
                                                                                                 parents_values)
        self.assertEqual(len(transition_scalar_indexing), len(parents_indxs) + 2)
        merged_list = parents_labels[:]
        merged_list.insert(0, node_id)
        merged_list.insert(0, node_id)
        vals_list = []
        for node_id in merged_list:
            vals_list.append(graph.get_states_number(node_id))
        m_vec = np.array([vals_list])
        m_vec = m_vec.cumprod()
        self.assertTrue(np.array_equal(transition_scalar_indexing, m_vec))

    def test_build_transition_scalar_indexing_structure(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        g1._aggregated_info_about_nodes_parents = g1.get_ord_set_of_par_of_all_nodes()
        p_labels = [i[0] for i in g1._aggregated_info_about_nodes_parents]
        p_vals = g1.get_ordered_by_indx_parents_values_for_all_nodes()
        fancy_indx = g1.build_fancy_indexing_structure(0)
        for node_id, p_i ,p_l, p_v in zip(g1._graph_struct.nodes_labels, fancy_indx, p_labels, p_vals):
            self.aux_build_transition_scalar_indexing_structure_for_a_node(g1, node_id, p_i ,p_l, p_v)

    def test_build_time_scalar_indexing_structure(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        g1._aggregated_info_about_nodes_parents = g1.get_ord_set_of_par_of_all_nodes()
        fancy_indx = g1.build_fancy_indexing_structure(0)
        p_labels = [i[0] for i in g1._aggregated_info_about_nodes_parents]
        p_vals = g1.get_ordered_by_indx_parents_values_for_all_nodes()
        #print(fancy_indx)
        for node_id, p_indxs, p_labels, p_v in zip(g1._graph_struct.nodes_labels, fancy_indx, p_labels, p_vals):
            self.aux_build_time_scalar_indexing_structure_for_a_node(g1, node_id, p_indxs, p_labels, p_v)

    def test_build_time_columns_filtering_structure(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        g1._aggregated_info_about_nodes_parents = g1.get_ord_set_of_par_of_all_nodes()
        g1._fancy_indexing = g1.build_fancy_indexing_structure(0)
        g1.build_time_columns_filtering_structure()
        t_filter = []
        for node_id, p_indxs in zip(g1.nodes, g1._fancy_indexing):
            single_filter = []
            single_filter.append(g1.get_node_indx(node_id))
            single_filter.extend(p_indxs)
            t_filter.append(np.array(single_filter))
        #print(t_filter)
        for a1, a2 in zip(g1.time_filtering, t_filter):
            self.assertTrue(np.array_equal(a1, a2))

    def test_build_transition_columns_filtering_structure(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        g1._aggregated_info_about_nodes_parents = g1.get_ord_set_of_par_of_all_nodes()
        g1._fancy_indexing = g1.build_fancy_indexing_structure(0)
        g1.build_transition_columns_filtering_structure()
        m_filter = []
        for node_id, p_indxs in zip(g1.nodes, g1._fancy_indexing):
            single_filter = []
            single_filter.append(g1.get_node_indx(node_id) + g1._graph_struct.total_variables_number)
            single_filter.append(g1.get_node_indx(node_id))
            single_filter.extend(p_indxs)
            m_filter.append(np.array(single_filter))
        for a1, a2 in zip(g1.transition_filtering, m_filter):
            self.assertTrue(np.array_equal(a1, a2))

    def test_build_p_combs_structure(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        g1._aggregated_info_about_nodes_parents = g1.get_ord_set_of_par_of_all_nodes()
        p_vals = g1.get_ordered_by_indx_parents_values_for_all_nodes()
        p_combs = g1.build_p_combs_structure()

        for matrix, p_v in zip(p_combs, p_vals):
            p_possible_vals = []
            for val in p_v:
                vals = [v for v in range(val)]
                p_possible_vals.extend(vals)
                comb_struct = set(itertools.product(p_possible_vals,repeat=len(p_v)))
                #print(comb_struct)
                for comb in comb_struct:
                    self.assertIn(np.array(comb), matrix)

    def test_fast_init(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g2 = ng.NetworkGraph(self.s1.structure)
        g1.init_graph()
        for indx, node in enumerate(g1.nodes):
            g2.fast_init(node)
            self.assertListEqual(g2._fancy_indexing, g1._fancy_indexing[indx])
            self.assertTrue(np.array_equal(g2.time_scalar_indexing_strucure, g1.time_scalar_indexing_strucure[indx]))
            self.assertTrue(np.array_equal(g2.transition_scalar_indexing_structure, g1.transition_scalar_indexing_structure[indx]))
            self.assertTrue(np.array_equal(g2.time_filtering, g1.time_filtering[indx]))
            self.assertTrue(np.array_equal(g2.transition_filtering, g1.transition_filtering[indx]))
            self.assertTrue(np.array_equal(g2.p_combs, g1.p_combs[indx]))

    def test_get_parents_by_id(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node in g1.nodes:
            self.assertListEqual(g1.get_parents_by_id(node), list(g1._graph.predecessors(node)))

    def test_get_states_number(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node, val in zip(g1.nodes, g1.nodes_values):
            self.assertEqual(val, g1.get_states_number(node))

    def test_get_node_indx(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node, indx in zip(g1.nodes, g1.nodes_indexes):
            self.assertEqual(indx, g1.get_node_indx(node))

if __name__ == '__main__':
    unittest.main()
