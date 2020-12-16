import sys
sys.path.append("../../classes/")
import glob
import math
import os
import unittest

import networkx as nx
import numpy as np
import psutil
from line_profiler import LineProfiler
import copy

import utility.cache as ch
import structure_graph.sample_path as sp
import estimators.structure_score_based_estimator as se
import estimators.structure_constraint_based_estimator as se_
import utility.json_importer as ji



class TestTabuSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass



    def test_constr(self):

        list_vals= [3,4,5,6,10,15,20]
        
        list_card=[[3,"ternary"],[4,"quaternary"]]

        list_dens = [["0.1","_01"],["0.2","_02"], ["0.3",""], ["0.4","_04"] ]
        
        for card in list_card:
            for dens in list_dens:

                list_vals= [3,4,5,6,10,15,20]
                
                if card[0]==4:
                    list_vals.remove(20)

                for var_n in list_vals:

                    patience = 20

                    var_number= var_n

                    if var_number > 11:
                        patience = 25

                    if var_number > 16:
                        patience = 35


                    cardinality = card[0]
                    cardinality_string = card[1]

                    density= dens[0]
                    density_string = dens[1]

                    constraint = 2

                    index = 0 
                    num_networks=10

                    if var_number > 9:
                        num_networks=3

                    while index < num_networks:
                        #cls.read_files = glob.glob(os.path.join('../../data', "*.json"))
                        self.importer = ji.JsonImporter(f"../../data/networks_and_trajectories_{cardinality_string}_data{density_string}_{var_number}.json", 
                                                    'samples', 'dyn.str', 'variables', 'Time', 'Name', index )
                        self.s1 = sp.SamplePath(self.importer)
                        self.s1.build_trajectories()
                        self.s1.build_structure()


                        true_edges = copy.deepcopy(self.s1.structure.edges)
                        true_edges = set(map(tuple, true_edges))
                        
                        s2= copy.deepcopy(self.s1)

                        se1 = se.StructureScoreBasedEstimator(self.s1,1,1)
                        edges_score = se1.estimate_structure(
                                                max_parents = None,
                                                iterations_number = 100,
                                                patience = patience,
                                                tabu_length = var_number,
                                                tabu_rules_duration = var_number,
                                                optimizer = 'tabu'
                                            )

                        se2 = se_.StructureConstraintBasedEstimator(s2, 0.1, 0.1)
                        edges_constraint = se2.estimate_structure()
                                        
                        set_list_edges = set.union(edges_constraint,edges_score)

            
                        n_added_fake_edges = len(set_list_edges.difference(true_edges))

                        n_missing_edges = len(true_edges.difference(set_list_edges))

                        n_true_positive = len(true_edges) - n_missing_edges

                        precision = n_true_positive / (n_true_positive + n_added_fake_edges)

                        recall = n_true_positive / (n_true_positive + n_missing_edges)

                        f1_measure = round(2* (precision*recall) / (precision+recall),3)

                        # print(f"n archi reali non trovati: {n_missing_edges}")
                        # print(f"n archi non reali aggiunti: {n_added_fake_edges}")
                        print(true_edges)
                        print(set_list_edges)
                        print(f"precision: {precision} ")
                        print(f"recall: {recall} ")


                        with open("../results/results.csv", 'a+') as fi:
                            fi.write(f"\n2,{constraint},{var_number},{density},{cardinality},{index},{f1_measure},{round(precision,3)},{round(recall,3)}")

                        index += 1

        self.assertEqual(set_list_edges, true_edges)
                


if __name__ == '__main__':
    unittest.main()

