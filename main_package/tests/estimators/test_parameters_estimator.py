
import unittest
import numpy as np
import glob
import os

from ...classes.structure_graph.network_graph import NetworkGraph
from ...classes.structure_graph.sample_path import SamplePath
from ...classes.structure_graph.set_of_cims import SetOfCims
from ...classes.estimators.parameters_estimator import ParametersEstimator
from ...classes.utility.json_importer import JsonImporter


class TestParametersEstimatior(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.read_files = glob.glob(os.path.join('./main_package/data/networks_and_trajectories_ternary_data_01_3.json', "*.json"))
        cls.array_indx = 0
        cls.importer = JsonImporter('./main_package/data/networks_and_trajectories_ternary_data_01_3.json', 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.importer.import_data(cls.array_indx)
        cls.s1 = SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()
        print(cls.s1.structure.edges)
        print(cls.s1.structure.nodes_values)

    def test_fast_init(self):
        for node in self.s1.structure.nodes_labels:
            g = NetworkGraph(self.s1.structure)
            g.fast_init(node)
            p1 = ParametersEstimator(self.s1.trajectories, g)
            self.assertEqual(p1._trajectories, self.s1.trajectories)
            self.assertEqual(p1._net_graph, g)
            self.assertIsNone(p1._single_set_of_cims)
            p1.fast_init(node)
            self.assertIsInstance(p1._single_set_of_cims, SetOfCims)

    def test_compute_parameters_for_node(self):
        for indx, node in enumerate(self.s1.structure.nodes_labels):
            print(node)
            g = NetworkGraph(self.s1.structure)
            g.fast_init(node)
            p1 = ParametersEstimator(self.s1.trajectories, g)
            p1.fast_init(node)
            sofc1 = p1.compute_parameters_for_node(node)
            sampled_cims = self.aux_import_sampled_cims('dyn.cims')
            sc = list(sampled_cims.values())
            self.equality_of_cims_of_node(sc[indx], sofc1._actual_cims)

    def equality_of_cims_of_node(self, sampled_cims, estimated_cims):
        self.assertEqual(len(sampled_cims), len(estimated_cims))
        for c1, c2 in zip(sampled_cims, estimated_cims):
            self.cim_equality_test(c1, c2.cim)

    def cim_equality_test(self, cim1, cim2):
        for r1, r2 in zip(cim1, cim2):
            self.assertTrue(np.all(np.isclose(r1, r2, 1e01)))

    def aux_import_sampled_cims(self, cims_label):
        i1 = JsonImporter('./main_package/data/networks_and_trajectories_ternary_data_01_3.json', '', '', '', '', '')
        raw_data = i1.read_json_file()
        return i1.import_sampled_cims(raw_data, self.array_indx, cims_label)


if __name__ == '__main__':
    unittest.main()
