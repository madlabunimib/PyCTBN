import pandas as pd
import numpy as np
import math
import itertools
import networkx as nx
from scipy.stats import f as f_dist
from scipy.stats import chi2 as chi2_dist


import sample_path as sp
import structure as st
import network_graph as ng
import parameters_estimator as pe


class StructureEstimator:

    def __init__(self, sample_path, exp_test_alfa, chi_test_alfa):
        self.sample_path = sample_path
        self.complete_graph_frame = self.build_complete_graph_frame(self.sample_path.structure.list_of_nodes_labels())
        self.complete_graph = self.build_complete_graph(self.sample_path.structure.list_of_nodes_labels())
        self.exp_test_sign = exp_test_alfa
        self.chi_test_alfa = chi_test_alfa

    def build_complete_graph_frame(self, node_ids):
        complete_frame = pd.DataFrame(itertools.permutations(node_ids, 2))
        complete_frame.columns = ['From', 'To']
        return complete_frame

    def build_complete_graph(self, node_ids):
        complete_graph = nx.DiGraph()
        complete_graph.add_nodes_from(node_ids)
        complete_graph.add_edges_from(itertools.permutations(node_ids, 2))
        return complete_graph

    def complete_test(self, test_parent, test_child, parent_set):
        tmp_df = self.complete_graph_frame.loc[self.complete_graph_frame['To'].isin([test_child])]
        #print(tmp_df)
        d1 = tmp_df.loc[tmp_df['From'].isin(parent_set)]
        parent_set.append(test_child)
        #print(parent_set)
        v1 = self.sample_path.structure.variables_frame.loc[self.sample_path.structure.variables_frame['Name'].isin(parent_set)]
        s1 = st.Structure(d1, v1, self.sample_path.total_variables_count)
        g1 = ng.NetworkGraph(s1)
        g1.init_graph()

        parent_set.append(test_parent)
        d2 = tmp_df.loc[tmp_df['From'].isin(parent_set)]
        v2 = self.sample_path.structure.variables_frame.loc[self.sample_path.structure.variables_frame['Name'].isin(parent_set)]
        #print(d2)
        #print(v2)
        s2 = st.Structure(d2, v2, self.sample_path.total_variables_count)
        g2 = ng.NetworkGraph(s2)
        g2.init_graph()

        p1 = pe.ParametersEstimator(self.sample_path, g1)
        p1.init_sets_cims_container()
        p1.compute_parameters()

        p2 = pe.ParametersEstimator(self.sample_path, g2)
        p2.init_sets_cims_container()
        p2.compute_parameters()

        #for cim in p1.sets_of_cims_struct.sets_of_cims[s1.get_positional_node_indx(test_child)].actual_cims:
            #print(cim)
            #print(cim.state_transition_matrix)
        #print("C_1", p1.sets_of_cims_struct.sets_of_cims[s1.get_positional_node_indx(test_child)].transition_matrices)
        indx = 0
        for i, cim1 in enumerate(
                p1.sets_of_cims_struct.sets_of_cims[s1.get_positional_node_indx(test_child)].actual_cims):

            #for j, cim2 in enumerate(
                    #p2.sets_of_cims_struct.sets_of_cims[s2.get_positional_node_indx(test_child)].actual_cims):
            for j in range(indx, self.sample_path.structure.get_states_number(test_parent) + indx):
                print("J", j)
                cim2 = p2.sets_of_cims_struct.sets_of_cims[s2.get_positional_node_indx(test_child)].actual_cims[j]
                indx += 1
                print(indx)


                print("Run Test", i, j)
                if not self.independence_test(test_child, cim1, cim2):
                    return False
        return True

    def independence_test(self, tested_child, cim1, cim2):
        # Fake exp test
        for val in range(0, self.sample_path.structure.get_states_number(tested_child)):  # i possibili valori di tested child TODO QUESTO CONTO DEVE ESSERE VETTORIZZATO
            r1 = cim1.state_transition_matrix[val][val]
            r2 = cim2.state_transition_matrix[val][val]
            print("No Test Parent:",cim1.cim[val][val],"With Test Parent", cim2.cim[val][val])
            F = cim2.cim[val][val] / cim1.cim[val][val]

            print("Exponential test", F, r1, r2)
            #print(f_dist.ppf(1 - self.exp_test_sign / 2, r1, r2))
            #print(f_dist.ppf(self.exp_test_sign / 2, r1, r2))
            if F < f_dist.ppf(self.exp_test_sign / 2, r1, r2) or \
                    F > f_dist.ppf(1 - self.exp_test_sign / 2, r1, r2):
                print("CONDITIONALLY DEPENDENT EXP")
                return False
        # fake chi test
        M1_no_diag = self.remove_diagonal_elements(cim1.state_transition_matrix)
        M2_no_diag = self.remove_diagonal_elements(cim2.state_transition_matrix)
        print("M1 no diag", M1_no_diag)
        print("M2 no diag", M2_no_diag)
        chi_2_quantile = chi2_dist.ppf(1 - self.chi_test_alfa, self.sample_path.structure.get_states_number(tested_child) - 1)
        for val in range(0, self.sample_path.structure.get_states_number(tested_child)):
            K = math.sqrt(cim1.state_transition_matrix[val][val] / cim2.state_transition_matrix[val][val])
            L = 1 / K
            Chi = np.sum(np.power(K * M2_no_diag[val] - L *M1_no_diag[val], 2) /
                         (M1_no_diag[val] + M2_no_diag[val]))
            print("Chi Stats", Chi)
            print("Chi Quantile", chi_2_quantile)
            if Chi > chi_2_quantile:
                print("CONDITIONALLY DEPENDENT CHI")
                return False
            #print("Chi test", Chi)
        return True

    def one_iteration_of_CTPC_algorithm(self, var_id):
        u = list(self.complete_graph.predecessors(var_id))
        tests_parents_numb = len(u)
        #print(u)
        b = 0
        parent_indx = 0
        while b < len(u):
            #for parent_id in u:
            parent_indx = 0
            while u and parent_indx < tests_parents_numb and b < len(u):
                # list_without_test_parent = u.remove(parent_id)
                removed = False
                print("b", b)
                print("Parent Indx", parent_indx)
                #if not list(self.generate_possible_sub_sets_of_size(u, b, u[parent_indx])):
                    #break
                S = self.generate_possible_sub_sets_of_size(u, b, u[parent_indx])
                print("U Set", u)
                print("S", S)
                for parents_set in S:
                    print("Parent Set", parents_set)
                    print("Test Parent", u[parent_indx])
                    if self.complete_test(u[parent_indx], var_id, parents_set):
                        print("Removing EDGE:", u[parent_indx], var_id)
                        self.complete_graph.remove_edge(u[parent_indx], var_id)
                        #self.complete_graph_frame = \
                            #self.complete_graph_frame[(self.complete_graph_frame.From !=
                                                     # u[parent_indx]) & (self.complete_graph_frame.To != var_id)]
                        u.remove(u[parent_indx])
                        removed = True
                    #else:
                        #parent_indx += 1
                if not removed:
                    parent_indx += 1
            b += 1

    def generate_possible_sub_sets_of_size(self, u, size, parent_id):
        print("Inside Generate subsets", u)
        print("InsideGenerate Subsets", parent_id)
        list_without_test_parent = u[:]
        list_without_test_parent.remove(parent_id)
        # u.remove(parent_id)
        #print(list(map(list, itertools.combinations(list_without_test_parent, size))))
        return map(list, itertools.combinations(list_without_test_parent, size))

    def remove_diagonal_elements(self, matrix):
        m = matrix.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = matrix.strides
        return strided(matrix.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)

