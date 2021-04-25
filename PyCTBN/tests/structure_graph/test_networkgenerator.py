import unittest
import random
import numpy as np

from PyCTBN.PyCTBN.structure_graph.network_generator import NetworkGenerator

class TestNetworkGenerator(unittest.TestCase):
    def test_generate_graph(self):
        labels = ["U", "V", "W", "X", "Y", "Z"]
        card = 3
        vals = [card for l in labels]
        ng = NetworkGenerator(labels, vals)
        ng.generate_graph(0.3)
        self.assertEqual(len(labels), len(ng.graph.nodes))
        self.assertEqual(len([edge for edge in ng.graph.edges if edge[0] == edge[1]]), 0)

    def test_generate_cims(self):
        labels = ["U", "V", "W", "X", "Y", "Z"]
        card = 3
        vals = [card for l in labels]
        cim_min = random.uniform(0.5, 5)
        cim_max = random.uniform(0.5, 5) + cim_min
        ng = NetworkGenerator(labels, vals)
        ng.generate_graph(0.3)
        ng.generate_cims(cim_min, cim_max)
        self.assertEqual(len(ng.cims), len(labels))      
        self.assertListEqual(list(ng.cims.keys()), labels)
        for key in ng.cims:
            p_card = ng.graph.get_ordered_by_indx_set_of_parents(key)[2]
            comb = ng.graph.build_p_comb_structure_for_a_node(p_card)
            self.assertEqual(len(ng.cims[key].actual_cims), len(comb))
            for cim in ng.cims[key].actual_cims:
                self.assertEqual(sum(c > 0 for c in cim.cim.diagonal()), 0)
                for i, row in enumerate(cim.cim):
                    self.assertEqual(round(sum(row) - row[i], 8), round(-1 * row[i], 8))
                    self.assertEqual(sum(c < 0 for c in np.delete(cim.cim[i], i)), 0)
    
unittest.main()