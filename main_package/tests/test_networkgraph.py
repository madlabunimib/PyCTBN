import unittest
import networkx as nx
import numpy as np
from line_profiler import LineProfiler

import sample_path as sp
import network_graph as ng


class TestNetworkGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s1 = sp.SamplePath('../data', 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.s1.build_trajectories()
        cls.s1.build_structure()

    def test_init(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        self.assertEqual(self.s1.structure, g1.graph_struct)
        self.assertIsInstance(g1.graph, nx.DiGraph)
        #TODO MANCANO TUTTI I TEST DI INIZIALIZZAZIONE DEI DATI PRIVATI della classe aggiungere le property necessarie

    def test_add_nodes(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.list_of_nodes_labels())
        for n1, n2 in zip(g1.get_nodes(), self.s1.structure.list_of_nodes_labels()):
            self.assertEqual(n1, n2)

    def test_add_edges(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_edges(self.s1.structure.list_of_edges())
        for e in self.s1.structure.list_of_edges():
            self.assertIn(tuple(e), g1.get_edges())

    """def test_get_ordered_by_indx_set_of_parents(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.list_of_nodes_labels())
        g1.add_edges(self.s1.structure.list_of_edges())
        sorted_par_list_aggregated_info = g1.get_ordered_by_indx_set_of_parents(g1.get_nodes()[2])
        self.test_aggregated_par_list_data(g1, g1.get_nodes()[2], sorted_par_list_aggregated_info)"""

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
                    self.assertEqual(graph.graph_struct.get_states_number(node), node_val)

    def test_get_ord_set_of_par_of_all_nodes(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.list_of_nodes_labels())
        g1.add_edges(self.s1.structure.list_of_edges())
        sorted_list_of_par_lists = g1.get_ord_set_of_par_of_all_nodes()
        for node, par_list in zip(g1.get_nodes_sorted_by_indx(), sorted_list_of_par_lists):
            self.aux_aggregated_par_list_data(g1, node, par_list)

    def test_get_ordered_by_indx_parents_values_for_all_nodes(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.list_of_nodes_labels())
        g1.add_edges(self.s1.structure.list_of_edges())
        g1.aggregated_info_about_nodes_parents = g1.get_ord_set_of_par_of_all_nodes()
        #print(g1.get_ordered_by_indx_parents_values_for_all_nodes())
        parents_values_list = g1.get_ordered_by_indx_parents_values_for_all_nodes()
        for pv1, aggr in zip(parents_values_list, g1.aggregated_info_about_nodes_parents):
            self.assertEqual(pv1, aggr[2])

    def test_get_states_number_of_all_nodes_sorted(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.list_of_nodes_labels())
        g1.add_edges(self.s1.structure.list_of_edges())
        nodes_cardinality_list = g1.get_states_number_of_all_nodes_sorted()
        for val, node in zip(nodes_cardinality_list, g1.get_nodes_sorted_by_indx()):
            self.assertEqual(val, g1.get_states_number(node))

    def test_build_fancy_indexing_structure_no_offset(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.list_of_nodes_labels())
        g1.add_edges(self.s1.structure.list_of_edges())
        g1.aggregated_info_about_nodes_parents = g1.get_ord_set_of_par_of_all_nodes()
        fancy_indx = g1.build_fancy_indexing_structure(0)
        for par_indxs, aggr in zip(fancy_indx, g1.aggregated_info_about_nodes_parents):
            self.assertEqual(par_indxs, aggr[1])

    def test_build_fancy_indexing_structure_offset(self):
        pass #TODO il codice di netgraph deve gestire questo caso

    def aux_build_time_scalar_indexing_structure_for_a_node(self, graph, node_indx, parents_indxs):
        time_scalar_indexing = graph.build_time_scalar_indexing_structure_for_a_node(node_indx, parents_indxs)
        self.assertEqual(len(time_scalar_indexing), len(parents_indxs) + 1)
        merged_list = parents_indxs[:]
        merged_list.insert(0, node_indx)
        #print(merged_list)
        vals_list = []
        for node in merged_list:
            vals_list.append(graph.get_states_number_by_indx(node))
        t_vec = np.array(vals_list)
        t_vec = t_vec.cumprod()
        #print(t_vec)
        self.assertTrue(np.array_equal(time_scalar_indexing, t_vec))

    def aux_build_transition_scalar_indexing_structure_for_a_node(self, graph, node_indx, parents_indxs):
        transition_scalar_indexing = graph.build_transition_scalar_indexing_structure_for_a_node(node_indx,
                                                                                                 parents_indxs)
        print(transition_scalar_indexing)
        self.assertEqual(len(transition_scalar_indexing), len(parents_indxs) + 2)
        merged_list = parents_indxs[:]
        merged_list.insert(0, node_indx)
        merged_list.insert(0, node_indx)
        vals_list = []
        for node in merged_list:
            vals_list.append(graph.get_states_number_by_indx(node))
        m_vec = np.array([vals_list])
        m_vec = m_vec.cumprod()
        self.assertTrue(np.array_equal(transition_scalar_indexing, m_vec))

    def test_build_transition_scalar_indexing_structure(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.list_of_nodes_labels())
        g1.add_edges(self.s1.structure.list_of_edges())
        g1.aggregated_info_about_nodes_parents = g1.get_ord_set_of_par_of_all_nodes()
        fancy_indx = g1.build_fancy_indexing_structure(0)
        print(fancy_indx)
        for node_id, p_indxs in zip(g1.graph_struct.list_of_nodes_indexes(), fancy_indx):
            self.aux_build_transition_scalar_indexing_structure_for_a_node(g1, node_id, p_indxs)

    def test_build_time_scalar_indexing_structure(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.list_of_nodes_labels())
        g1.add_edges(self.s1.structure.list_of_edges())
        g1.aggregated_info_about_nodes_parents = g1.get_ord_set_of_par_of_all_nodes()
        fancy_indx = g1.build_fancy_indexing_structure(0)
        #print(fancy_indx)
        for node_id, p_indxs in zip(g1.graph_struct.list_of_nodes_indexes(), fancy_indx):
            self.aux_build_time_scalar_indexing_structure_for_a_node(g1, node_id, p_indxs)

    def test_build_time_columns_filtering_structure(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.list_of_nodes_labels())
        g1.add_edges(self.s1.structure.list_of_edges())
        g1.aggregated_info_about_nodes_parents = g1.get_ord_set_of_par_of_all_nodes()
        g1._fancy_indexing = g1.build_fancy_indexing_structure(0)
        g1.build_time_columns_filtering_structure()
        print(g1.time_filtering)
        t_filter = []
        for node_id, p_indxs in zip(g1.get_nodes_sorted_by_indx(), g1._fancy_indexing):
            single_filter = []
            single_filter.append(g1.get_node_indx(node_id))
            single_filter.extend(p_indxs)
            t_filter.append(np.array(single_filter))
        #print(t_filter)
        for a1, a2 in zip(g1.time_filtering, t_filter):
            self.assertTrue(np.array_equal(a1, a2))

    def test_build_transition_columns_filtering_structure(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.add_nodes(self.s1.structure.list_of_nodes_labels())
        g1.add_edges(self.s1.structure.list_of_edges())
        g1.aggregated_info_about_nodes_parents = g1.get_ord_set_of_par_of_all_nodes()
        g1._fancy_indexing = g1.build_fancy_indexing_structure(0)
        g1.build_transition_columns_filtering_structure()
        print(g1.transition_filtering)
        m_filter = []
        for node_id, p_indxs in zip(g1.get_nodes_sorted_by_indx(), g1._fancy_indexing):
            single_filter = []
            single_filter.append(g1.get_node_indx(node_id) + g1.graph_struct.total_variables_number)
            single_filter.append(g1.get_node_indx(node_id))
            single_filter.extend(p_indxs)
            m_filter.append(np.array(single_filter))
        print(m_filter)
        for a1, a2 in zip(g1.transition_filtering, m_filter):
            self.assertTrue(np.array_equal(a1, a2))

    def test_init_graph(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        #g1.build_scalar_indexing_structures()
        lp = LineProfiler()
        #lp.add_function(g1.get_ordered_by_indx_set_of_parents)
        #lp.add_function(g1.get_states_number)
        lp_wrapper = lp(g1.init_graph)
        print(g1.time_scalar_indexing_strucure)
        print(g1.transition_scalar_indexing_structure)
        """[array([3]), array([3, 9]), array([ 3,  9, 27])]
[array([3, 9]), array([ 3,  9, 27]), array([ 3,  9, 27, 81])]"""
        lp_wrapper()
        lp.print_stats()

    """def test_remove_node(self):
        g1 = ng.NetworkGraph(self.s1.structure)
        g1.init_graph()
        g1.remove_node('Y')
        print(g1.get_nodes())
        print(g1.get_edges())"""




#TODO mancano i test sulle property e sui getters_vari
if __name__ == '__main__':
    unittest.main()
