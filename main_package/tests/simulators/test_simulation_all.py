import sys
sys.path.append("../../classes/")
import glob
import math
import os
import unittest

import networkx as nx
import numpy as np
import pandas as pd
import psutil
from line_profiler import LineProfiler
import copy
import json

import utility.cache as ch
import structure_graph.sample_path as sp
import estimators.structure_score_based_estimator as se_score
import estimators.structure_constraint_based_estimator as se_constr
import utility.sample_importer as si



class TestTabuSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass



    def test_constr(self):

        list_constraint= [0,1]
        
        list_cardinality= [[2,"binary"],[3,"ternary"], [4,"quaternary"]]

        
        
        list_dens = [["0.1","_01"],["0.2","_02"], ["0.3",""], ["0.4","_04"] ]

        for constr in list_constraint:
            for card in list_cardinality:
                for dens in list_dens:
                    
                    if card[0] == 4:
                        list_vals= [3,4,5,6,10,15]
                    else:
                        list_vals= [3,4,5,6,10,15,20]

                    for var_n in list_vals:

                        patience = 25

                        var_number= var_n

                        if var_number > 11:
                            patience = 30

                        if var_number > 16:
                            patience = 35


                        cardinality = card[0]
                        cardinality_string = card[1]

                        density= dens[0]
                        density_string = dens[1]

                        constraint = constr

                        index = 1
                        num_networks=10


                        while index <= num_networks:

                            with open(f"/home/alessandro/Documents/ctbn_cba/data/networks_and_trajectories_{cardinality_string}_data{density_string}_{var_number}/{index}.json") as f:
                                raw_data = json.load(f)

                                trajectory_list_raw= raw_data["samples"]

                                trajectory_list = [pd.DataFrame(sample) for sample in trajectory_list_raw]

                                variables= pd.DataFrame(raw_data["variables"])
                                prior_net_structure = pd.DataFrame(raw_data["dyn.str"])


                            self.importer = si.SampleImporter(
                                                            trajectory_list=trajectory_list,
                                                            variables=variables,
                                                            prior_net_structure=prior_net_structure
                                                        )
                            
                            self.importer.import_data()
                            self.s1 = sp.SamplePath(self.importer)
                            self.s1.build_trajectories()
                            self.s1.build_structure()


                            true_edges = copy.deepcopy(self.s1.structure.edges)
                            true_edges = set(map(tuple, true_edges))

                            if constr == 0:
                                se1 = se_score.StructureScoreBasedEstimator(self.s1)
                                set_list_edges = se1.estimate_structure(
                                                    max_parents = None,
                                                    iterations_number = 100,
                                                    patience = patience,
                                                    tabu_length = var_number,
                                                    tabu_rules_duration = var_number,
                                                    optimizer = 'tabu'
                                                    )
                            else:
                                se1 = se_constr.StructureConstraintBasedEstimator(self.s1,0.1,0.1)
                                set_list_edges = se1.estimate_structure(disable_multiprocessing=False)
               
                            n_added_fake_edges = len(set_list_edges.difference(true_edges))

                            n_missing_edges = len(true_edges.difference(set_list_edges))

                            n_true_positive = len(true_edges) - n_missing_edges

                            precision = n_true_positive / (n_true_positive + n_added_fake_edges)

                            recall = n_true_positive / (n_true_positive + n_missing_edges)

                            f1_measure = round(2* (precision*recall) / (precision+recall),3)

                            print(true_edges)
                            print(set_list_edges)
                            print(f"precision: {precision} ")
                            print(f"recall: {recall} ")

                            with open("../results/results.csv", 'a+') as fi:
                                fi.write(f"{constraint},{var_number},{density},{cardinality},{index},{f1_measure},{round(precision,3)},{round(recall,3)}")

                            index += 1

        self.assertEqual(set_list_edges, true_edges)
                
                


if __name__ == '__main__':
    unittest.main()

