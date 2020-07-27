import unittest
import numpy as np
from line_profiler import LineProfiler

import network_graph as ng
import sample_path as sp
import sets_of_cims_container as scc
import parameters_estimator as pe
import json_importer as ji

#TODO bisogna trovare un modo per testare i metodi che stimano i tempi e le transizioni per i singoli nodi
class TestParametersEstimatior(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.s1 = sp.SamplePath('../data', 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        cls.s1.build_trajectories()
        cls.s1.build_structure()
        cls.g1 = ng.NetworkGraph(cls.s1.structure)
        cls.g1.init_graph()

    def test_init(self):
        self.aux_test_init(self.s1, self.g1)

    def test_init_sets_of_cims_container(self):
        self.aux_test_init_sets_cims_container(self.s1, self.g1)

    def aux_test_init(self, sample_p, graph):
        pe1 = pe.ParametersEstimator(sample_p, graph)
        self.assertEqual(sample_p, pe1.sample_path)
        self.assertEqual(graph, pe1.net_graph)
        self.assertIsNone(pe1.sets_of_cims_struct)

    def aux_test_init_sets_cims_container(self, sample_p, graph):
        pe1 = pe.ParametersEstimator(sample_p, graph)
        pe1.init_sets_cims_container()
        self.assertIsInstance(pe1.sets_of_cims_struct, scc.SetsOfCimsContainer)

    def test_compute_parameters(self):
        self.aux_test_compute_parameters(self.s1, self.g1)

    def aux_test_compute_parameters(self, sample_p, graph):
        pe1 = pe.ParametersEstimator(sample_p, graph)
        pe1.init_sets_cims_container()
        pe1.compute_parameters()
        samples_cims = self.aux_import_sampled_cims('dyn.cims')
        for indx, sc in enumerate(samples_cims.values()):
            self.equality_of_cims_of_node(sc, pe1.sets_of_cims_struct.get_set_of_cims(indx).get_cims())

    def equality_of_cims_of_node(self, sampled_cims, estimated_cims):
        self.assertEqual(len(sampled_cims), len(estimated_cims))
        for c1, c2 in zip(sampled_cims, estimated_cims):
            self.cim_equality_test(c1, c2.cim)

    def cim_equality_test(self, cim1, cim2):
        for r1, r2 in zip(cim1, cim2):
            self.assertTrue(np.all(np.isclose(r1, r2, 1e-01, 1e-01) == True))

    def test_compute_parameters_for_node(self):#TODO Questo non è un test
        self.g1.remove_node('Y')
        print(self.g1.time_filtering)
        pe1 = pe.ParametersEstimator(self.s1, self.g1)
        pe1.init_sets_cims_container()
        lp = LineProfiler()
        #lp.add_function(pe1.init_sets_cims_container)
        lp_wrapper = lp(pe1.compute_parameters_for_node)
        lp_wrapper('X')
        lp.print_stats()
        #pe1.init_sets_cims_container()
        #pe1.compute_parameters_for_node('Y')
        print(pe1.sets_of_cims_struct.get_set_of_cims(0).actual_cims)


    def aux_import_sampled_cims(self, cims_label):
        i1 = ji.JsonImporter('../data', '', '', '', '', '')
        raw_data = i1.read_json_file()
        return i1.import_sampled_cims(raw_data, 0, cims_label)





if __name__ == '__main__':
    unittest.main()
