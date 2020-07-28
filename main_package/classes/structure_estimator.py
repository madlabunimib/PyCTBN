import pandas as pd
import numpy as np
import itertools
import networkx as nx
from scipy.stats import f as f_dist
from scipy.stats import chi2 as chi2_dist




import sample_path as sp
import structure as st
import network_graph as ng
import parameters_estimator as pe
import cache as ch


class StructureEstimator:

    def __init__(self, sample_path, exp_test_alfa, chi_test_alfa):
        self.sample_path = sample_path
        self.complete_graph_frame = self.build_complete_graph_frame(self.sample_path.structure.list_of_nodes_labels())
        self.complete_graph = self.build_complete_graph(self.sample_path.structure.list_of_nodes_labels())
        self.exp_test_sign = exp_test_alfa
        self.chi_test_alfa = chi_test_alfa
        self.cache = ch.Cache()

    def build_complete_graph_frame(self, node_ids):
        complete_frame = pd.DataFrame(itertools.permutations(node_ids, 2))
        complete_frame.columns = ['From', 'To']
        return complete_frame

    def build_complete_graph(self, node_ids):
        complete_graph = nx.DiGraph()
        complete_graph.add_nodes_from(node_ids)
        complete_graph.add_edges_from(itertools.permutations(node_ids, 2))
        return complete_graph

    def complete_test(self, tmp_df, test_parent, test_child, parent_set):
        p_set = parent_set[:]
        complete_info = parent_set[:]
        complete_info.append(test_parent)
        #tmp_df = self.complete_graph_frame.loc[self.complete_graph_frame['To'].isin([test_child])]
        #tmp_df = self.complete_graph_frame.loc[np.in1d(self.complete_graph_frame['To'], test_child)]
        d2 = tmp_df.loc[tmp_df['From'].isin(complete_info)]
        complete_info.append(test_child)
        values_frame = self.sample_path.structure.variables_frame
        v2 = values_frame.loc[
                values_frame['Name'].isin(complete_info)]

        #print(tmp_df)
        #d1 = tmp_df.loc[tmp_df['From'].isin(parent_set)]
        #parent_set.append(test_child)
        #print(parent_set)
        """v1 = self.sample_path.structure.variables_frame.loc[self.sample_path.structure.variables_frame['Name'].isin(parent_set)]
        s1 = st.Structure(d1, v1, self.sample_path.total_variables_count)
        g1 = ng.NetworkGraph(s1)
        g1.init_graph()"""

        #parent_set.append(test_parent)
        """d2 = tmp_df.loc[tmp_df['From'].isin(parent_set)]
        v2 = self.sample_path.structure.variables_frame.loc[self.sample_path.structure.variables_frame['Name'].isin(parent_set)]
        s2 = st.Structure(d2, v2, self.sample_path.total_variables_count)
        g2 = ng.NetworkGraph(s2)
        g2.init_graph()"""
        #parent_set.append(test_child)
        #sofc1 = None
        #if not sofc1:
        if not p_set:
            sofc1 = self.cache.find(test_child)
        else:
            sofc1 = self.cache.find(set(p_set))

        if not sofc1:
            #d1 = tmp_df.loc[tmp_df['From'].isin(parent_set)]
            d1 = d2[d2.From != test_parent]

            #v1 = self.sample_path.structure.variables_frame.loc[
                #self.sample_path.structure.variables_frame['Name'].isin(parent_set)]
            v1 = v2[v2.Name != test_parent]
            #print("D1", d1)
            #print("V1", v1)
            s1 = st.Structure(d1, v1, self.sample_path.total_variables_count)
            g1 = ng.NetworkGraph(s1)
            g1.init_graph()
            p1 = pe.ParametersEstimator(self.sample_path, g1)
            p1.init_sets_cims_container()
            p1.compute_parameters_for_node(test_child)
            sofc1 = p1.sets_of_cims_struct.sets_of_cims[s1.get_positional_node_indx(test_child)]
            if not p_set:
                self.cache.put(test_child, sofc1)
            else:
                self.cache.put(set(p_set), sofc1)
        sofc2 = None
        p_set.append(test_parent)
        if p_set:
            #p_set.append(test_parent)
            #print("PSET ", p_set)
            #set_p_set = set(p_set)
            sofc2 = self.cache.find(set(p_set))
            #print("Sofc2 ", sofc2)
            #print(self.cache.list_of_sets_of_indxs)

        """p2 = pe.ParametersEstimator(self.sample_path, g2)
        p2.init_sets_cims_container()
        #p2.compute_parameters()
        p2.compute_parameters_for_node(test_child)
        sofc2 = p2.sets_of_cims_struct.sets_of_cims[s2.get_positional_node_indx(test_child)]"""
        if not sofc2:
            print("Cache Miss SOC2")
            #parent_set.append(test_parent)
            #d2 = tmp_df.loc[tmp_df['From'].isin(p_set)]
            #v2 = self.sample_path.structure.variables_frame.loc[
                #self.sample_path.structure.variables_frame['Name'].isin(parent_set)]
            #print("D2", d2)
            #print("V2", v2)
            #s2 = st.Structure(d2, v2, self.sample_path.total_variables_count)
            s2 = st.Structure(d2, v2, self.sample_path.total_variables_count)
            g2 = ng.NetworkGraph(s2)
            g2.init_graph()
            p2 = pe.ParametersEstimator(self.sample_path, g2)
            p2.init_sets_cims_container()
            p2.compute_parameters_for_node(test_child)
            sofc2 = p2.sets_of_cims_struct.sets_of_cims[s2.get_positional_node_indx(test_child)]
            if p_set:
                #set_p_set = set(p_set)
                self.cache.put(set(p_set), sofc2)
        end = 0
        increment = self.sample_path.structure.get_states_number(test_parent)
        for cim1 in sofc1.actual_cims:
            start = end
            end = start + increment
            for j in range(start, end):
                #cim2 = sofc2.actual_cims[j]
                #print(indx)
                #print("Run Test", i, j)
                if not self.independence_test(test_child, cim1, sofc2.actual_cims[j]):
                    return False
        return True

    def independence_test(self, tested_child, cim1, cim2):
        r1s = cim1.state_transition_matrix.diagonal()
        r2s = cim2.state_transition_matrix.diagonal()
        F_stats = cim2.cim.diagonal() / cim1.cim.diagonal()
        child_states_numb = self.sample_path.structure.get_states_number(tested_child)
        for val in range(0, child_states_numb):
            if F_stats[val] < f_dist.ppf(self.exp_test_sign / 2, r1s[val], r2s[val]) or \
                    F_stats[val] > f_dist.ppf(1 - self.exp_test_sign / 2, r1s[val], r2s[val]):
                print("CONDITIONALLY DEPENDENT EXP")
                return False
        #M1_no_diag = self.remove_diagonal_elements(cim1.state_transition_matrix)
        #M2_no_diag = self.remove_diagonal_elements(cim2.state_transition_matrix)
        M1_no_diag = cim1.state_transition_matrix[~np.eye(cim1.state_transition_matrix.shape[0], dtype=bool)].reshape(cim1.state_transition_matrix.shape[0], -1)
        M2_no_diag = cim2.state_transition_matrix[~np.eye(cim2.state_transition_matrix.shape[0], dtype=bool)].reshape(
            cim2.state_transition_matrix.shape[0], -1)
        chi_2_quantile = chi2_dist.ppf(1 - self.chi_test_alfa, child_states_numb - 1)
        """
        Ks = np.sqrt(cim1.state_transition_matrix.diagonal() / cim2.state_transition_matrix.diagonal())
        Ls = np.reciprocal(Ks)
        chi_stats = np.sum((np.power((M2_no_diag.T * Ks).T - (M1_no_diag.T * Ls).T, 2) \
                            / (M1_no_diag + M2_no_diag)), axis=1)"""
        Ks = np.sqrt(r1s / r2s)
        Ls = np.sqrt(r2s / r1s)
        for val in range(0, child_states_numb):
            #K = math.sqrt(cim1.state_transition_matrix[val][val] / cim2.state_transition_matrix[val][val])
            #L = 1 / K
            Chi = np.sum(np.power(Ks[val] * M2_no_diag[val] - Ls[val] *M1_no_diag[val], 2) /
                         (M1_no_diag[val] + M2_no_diag[val]))

            #print("Chi Stats", Chi)
            #print("Chi Quantile", chi_2_quantile)
            if Chi > chi_2_quantile:
        #if np.any(chi_stats > chi_2_quantile):
                print("CONDITIONALLY DEPENDENT CHI")
                return False
            #print("Chi test", Chi)
        return True

    def one_iteration_of_CTPC_algorithm(self, var_id):
        u = list(self.complete_graph.predecessors(var_id))
        tests_parents_numb = len(u)
        complete_frame = self.complete_graph_frame
        test_frame = complete_frame.loc[complete_frame['To'].isin([var_id])]
        b = 0
        while b < len(u):
            #for parent_id in u:
            parent_indx = 0
            while u and parent_indx < tests_parents_numb and b < len(u):
                # list_without_test_parent = u.remove(parent_id)
                removed = False
                #print("b", b)
                #print("Parent Indx", parent_indx)
                #if not list(self.generate_possible_sub_sets_of_size(u, b, u[parent_indx])):
                    #break
                S = self.generate_possible_sub_sets_of_size(u, b, parent_indx)
                #print("U Set", u)
                #print("S", S)
                for parents_set in S:
                    #print("Parent Set", parents_set)
                    #print("Test Parent", u[parent_indx])
                    if self.complete_test(test_frame, u[parent_indx], var_id, parents_set):
                        #print("Removing EDGE:", u[parent_indx], var_id)
                        self.complete_graph.remove_edge(u[parent_indx], var_id)
                        #print(self.complete_graph_frame)
                        """self.complete_graph_frame = \
                            self.complete_graph_frame.drop(
                                self.complete_graph_frame[(self.complete_graph_frame.From ==
                                                     u[parent_indx]) & (self.complete_graph_frame.To == var_id)].index)"""

                        complete_frame.drop(complete_frame[(complete_frame.From == u[parent_indx]) &
                                                           (complete_frame.To == var_id)].index, inplace=True)
                        #print(self.complete_graph_frame)
                        #u.remove(u[parent_indx])
                        del u[parent_indx]
                        removed = True
                    #else:
                        #parent_indx += 1
                if not removed:
                    parent_indx += 1
            b += 1
        self.cache.clear()

    def generate_possible_sub_sets_of_size(self, u, size, parent_indx):
        #print("Inside Generate subsets", u)
        #print("InsideGenerate Subsets", parent_id)
        list_without_test_parent = u[:]
        del list_without_test_parent[parent_indx]
        # u.remove(parent_id)
        #print(list(map(list, itertools.combinations(list_without_test_parent, size))))
        return map(list, itertools.combinations(list_without_test_parent, size))

    def remove_diagonal_elements(self, matrix):
        m = matrix.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = matrix.strides
        return strided(matrix.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)

    def ctpc_algorithm(self):
        ctpc_algo = self.one_iteration_of_CTPC_algorithm
        nodes = self.sample_path.structure.list_of_nodes_labels()
        #for node_id in self.sample_path.structure.list_of_nodes_labels():
            #print("TESTING VAR:", node_id)
            #self.one_iteration_of_CTPC_algorithm(node_id)
            #print(self.complete_graph_frame)
        [ctpc_algo(n) for n in nodes]

