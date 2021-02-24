
import unittest
import glob
import os
import networkx as nx
import numpy as np
import itertools

from ...classes.structure_graph.sample_path import SamplePath
from ...classes.structure_graph.network_graph import NetworkGraph
from ...classes.utility.json_importer import JsonImporter


class TestNetworkGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.read_files = glob.glob(os.path.join('./main_package/data', "*.json"))
        cls.importer = JsonImporter(cls.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.importer.import_data(0)
        cls.s1 = SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()

    def test_init(self):
        g1 = NetworkGraph(self.s1.structure)
        self.assertEqual(self.s1.structure, g1._graph_struct)
        self.assertIsInstance(g1._graph, nx.DiGraph)
        self.assertIsNone(g1.time_scalar_indexing_strucure)
        self.assertIsNone(g1.transition_scalar_indexing_structure)
        self.assertIsNone(g1.transition_filtering)
        self.assertIsNone(g1.p_combs)

    def test_add_nodes(self):
        g1 = NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        for n1, n2 in zip(g1.nodes, self.s1.structure.nodes_labels):
            self.assertEqual(n1, n2)

    def test_add_edges(self):
        g1 = NetworkGraph(self.s1.structure)
        g1.add_edges(self.s1.structure.edges)
        for e in self.s1.structure.edges:
            self.assertIn(tuple(e), g1.edges)

    def test_fast_init(self):
        g1 = NetworkGraph(self.s1.structure)
        for node in self.s1.structure.nodes_labels:
            g1.fast_init(node)
            self.assertIsNotNone(g1._graph.nodes)
            self.assertIsNotNone(g1._graph.edges)
            self.assertIsInstance(g1._time_scalar_indexing_structure, np.ndarray)
            self.assertIsInstance(g1._transition_scalar_indexing_structure, np.ndarray)
            self.assertIsInstance(g1._time_filtering, np.ndarray)
            self.assertIsInstance(g1._transition_filtering, np.ndarray)
            self.assertIsInstance(g1._p_combs_structure, np.ndarray)
            self.assertIsInstance(g1._aggregated_info_about_nodes_parents, tuple)

    def test_get_ordered_by_indx_set_of_parents(self):
        g1 = NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node in self.s1.structure.nodes_labels:
            aggr_info = g1.get_ordered_by_indx_set_of_parents(node)
            for indx in range(len(aggr_info[0]) - 1 ):
                self.assertLess(g1.get_node_indx(aggr_info[0][indx]), g1.get_node_indx(aggr_info[0][indx + 1]))
            for par, par_indx in zip(aggr_info[0], aggr_info[1]):
                self.assertEqual(g1.get_node_indx(par), par_indx)
            for par, par_val in zip(aggr_info[0], aggr_info[2]):
                self.assertEqual(g1._graph_struct.get_states_number(par), par_val)

    def test_build_time_scalar_indexing_structure_for_a_node(self):
        g1 = NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node in self.s1.structure.nodes_labels:
            aggr_info = g1.get_ordered_by_indx_set_of_parents(node)
            self.aux_build_time_scalar_indexing_structure_for_a_node(g1, node, aggr_info[1],
                                                                     aggr_info[0], aggr_info[2])

    def aux_build_time_scalar_indexing_structure_for_a_node(self, graph, node_id, parents_indxs, parents_labels, parents_vals):
        node_states = graph.get_states_number(node_id)
        time_scalar_indexing = NetworkGraph.build_time_scalar_indexing_structure_for_a_node(node_states, parents_vals)
        self.assertEqual(len(time_scalar_indexing), len(parents_indxs) + 1)
        merged_list = parents_labels[:]
        merged_list.insert(0, node_id)
        vals_list = []
        for node in merged_list:
            vals_list.append(graph.get_states_number(node))
        t_vec = np.array(vals_list)
        t_vec = t_vec.cumprod()
        self.assertTrue(np.array_equal(time_scalar_indexing, t_vec))

    def test_build_transition_scalar_indexing_structure_for_a_node(self):
        g1 = NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node in self.s1.structure.nodes_labels:
            aggr_info = g1.get_ordered_by_indx_set_of_parents(node)
            self.aux_build_transition_scalar_indexing_structure_for_a_node(g1, node, aggr_info[1],
                                                                        aggr_info[0], aggr_info[2])

    def aux_build_transition_scalar_indexing_structure_for_a_node(self, graph, node_id, parents_indxs, parents_labels,
                                                                  parents_values):
        node_states = graph.get_states_number(node_id)
        transition_scalar_indexing = graph.build_transition_scalar_indexing_structure_for_a_node(node_states,
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

    def test_build_time_columns_filtering_structure_for_a_node(self):
        g1 = NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node in self.s1.structure.nodes_labels:
            aggr_info = g1.get_ordered_by_indx_set_of_parents(node)
            self.aux_build_time_columns_filtering_structure_for_a_node(g1, node, aggr_info[1])

    def aux_build_time_columns_filtering_structure_for_a_node(self, graph, node_id, p_indxs):
        graph.build_time_columns_filtering_for_a_node(graph.get_node_indx(node_id), p_indxs)
        single_filter = []
        single_filter.append(graph.get_node_indx(node_id))
        single_filter.extend(p_indxs)
        self.assertTrue(np.array_equal(graph.build_time_columns_filtering_for_a_node(graph.get_node_indx(node_id),
                                                                                     p_indxs),np.array(single_filter)))
    def test_build_transition_columns_filtering_structure(self):
        g1 = NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node in self.s1.structure.nodes_labels:
            aggr_info = g1.get_ordered_by_indx_set_of_parents(node)
            self.aux_build_time_columns_filtering_structure_for_a_node(g1, node, aggr_info[1])

    def aux_build_transition_columns_filtering_structure(self, graph, node_id, p_indxs):
        single_filter = []
        single_filter.append(graph.get_node_indx(node_id) + graph._graph_struct.total_variables_number)
        single_filter.append(graph.get_node_indx(node_id))
        single_filter.extend(p_indxs)
        self.assertTrue(np.array_equal(graph.build_transition_filtering_for_a_node(graph.get_node_indx(node_id),

                                                                                     p_indxs), np.array(single_filter)))
    def test_build_p_combs_structure(self):
        g1 = NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node in self.s1.structure.nodes_labels:
            aggr_info = g1.get_ordered_by_indx_set_of_parents(node)
            self.aux_build_p_combs_structure(g1, aggr_info[2])

    def aux_build_p_combs_structure(self, graph, p_vals):
        p_combs = graph.build_p_comb_structure_for_a_node(p_vals)
        p_possible_vals = []
        for val in p_vals:
            vals = [v for v in range(val)]
            p_possible_vals.extend(vals)
            comb_struct = set(itertools.product(p_possible_vals,repeat=len(p_vals)))
            for comb in comb_struct:
                self.assertIn(np.array(comb), p_combs)

    def test_get_parents_by_id(self):
        g1 = NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node in g1.nodes:
            self.assertListEqual(g1.get_parents_by_id(node), list(g1._graph.predecessors(node)))

    def test_get_states_number(self):
        g1 = NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node, val in zip(g1.nodes, g1.nodes_values):
            self.assertEqual(val, g1.get_states_number(node))

    def test_get_node_indx(self):
        g1 = NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.nodes_labels)
        g1.add_edges(self.s1.structure.edges)
        for node, indx in zip(g1.nodes, g1.nodes_indexes):
            self.assertEqual(indx, g1.get_node_indx(node))


if __name__ == '__main__':
    unittest.main()
