import sys
sys.path.append("../classes/")
import unittest
import numpy as np
import glob
import os

import network_graph as ng
import sample_path as sp
import set_of_cims as sofc
import parameters_estimator as pe
import json_importer as ji


class TestParametersEstimatior(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.read_files = glob.glob(os.path.join('../data', "*.json"))
        cls.array_indx = 8
        cls.importer = ji.JsonImporter(cls.read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name',
                                       cls.array_indx)
        cls.s1 = sp.SamplePath(cls.importer)
        cls.s1.build_trajectories()
        cls.s1.build_structure()
        print(cls.s1.structure.edges)
        print(cls.s1.structure.nodes_values)
        cls.g1 = ng.NetworkGraph(cls.s1.structure)
        cls.g1.init_graph()

    def test_fast_init(self):
        for node in self.g1.nodes:
            g = ng.NetworkGraph(self.s1.structure)
            g.fast_init(node)
            p1 = pe.ParametersEstimator(self.s1.trajectories, g)
            self.assertEqual(p1._trajectories, self.s1.trajectories)
            self.assertEqual(p1._net_graph, g)
            self.assertIsNone(p1._single_set_of_cims)
            p1.fast_init(node)
            self.assertIsInstance(p1._single_set_of_cims, sofc.SetOfCims)

    def test_compute_parameters_for_node(self):
        for indx, node in enumerate(self.g1.nodes):
            print(node)
            g = ng.NetworkGraph(self.s1.structure)
            g.fast_init(node)
            p1 = pe.ParametersEstimator(self.s1.trajectories, g)
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
            self.assertTrue(np.all(np.isclose(r1, r2, 1e-01, 1e-01) == True))

    def aux_import_sampled_cims(self, cims_label):
        i1 = ji.JsonImporter(self.read_files[0], '', '', '', '', '', self.array_indx)
        raw_data = i1.read_json_file()
        return i1.import_sampled_cims(raw_data, self.array_indx, cims_label)

if __name__ == '__main__':
    unittest.main()
