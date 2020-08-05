
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
    """
    Has the task of estimating the network structure given the trajectories in samplepath.

    :sample_path: the sample_path object containing the trajectories and the real structure
    :exp_test_sign: the significance level for the exponential Hp test
    :chi_test_alfa: the significance level for the chi Hp test

    :nodes: the nodes labels
    :nodes_vals: the nodes cardinalities
    :nodes_indxs: the nodes indexes
    :complete_graph: the complete directed graph built using the nodes labels in nodes
    :cache: the cache object
    """

    def __init__(self, sample_path: sp.SamplePath, exp_test_alfa: float, chi_test_alfa: float):
        self.sample_path = sample_path
        self.nodes = np.array(self.sample_path.structure.nodes_labels)
        self.nodes_vals = self.sample_path.structure.nodes_values
        self.nodes_indxs = self.sample_path.structure.nodes_indexes
        self.complete_graph = self.build_complete_graph(self.sample_path.structure.nodes_labels)
        self.exp_test_sign = exp_test_alfa
        self.chi_test_alfa = chi_test_alfa
        self.cache = ch.Cache()

    def build_complete_graph(self, node_ids):
        complete_graph = nx.DiGraph()
        complete_graph.add_nodes_from(node_ids)
        complete_graph.add_edges_from(itertools.permutations(node_ids, 2))
        return complete_graph

    def complete_test(self, test_parent, test_child, parent_set, child_states_numb, tot_vars_count):
        #print("Test Parent:", test_parent)
        #print("Sep Set", parent_set)
        p_set = parent_set[:]
        complete_info = parent_set[:]
        complete_info.append(test_child)

        parents = np.array(parent_set)
        parents = np.append(parents, test_parent)
        #print("PARENTS", parents)
        #parents.sort()
        sorted_parents = self.nodes[np.isin(self.nodes, parents)]
        #print("SORTED PARENTS", sorted_parents)
        cims_filter = sorted_parents != test_parent
        #print("PARENTS NO FROM MASK", cims_filter)
        if not p_set:
            #print("EMPTY PSET TRYING TO FIND", test_child)
            sofc1 = self.cache.find(test_child)
        else:
            sofc1 = self.cache.find(set(p_set))

        if not sofc1:
            #print("CACHE MISSS SOFC1")
            bool_mask1 = np.isin(self.nodes,complete_info)
            #print("Bool mask 1", bool_mask1)
            l1 = list(self.nodes[bool_mask1])
            #print("L1", l1)
            indxs1 = self.nodes_indxs[bool_mask1]
            #print("INDXS 1", indxs1)
            vals1 = self.nodes_vals[bool_mask1]
            eds1 = list(itertools.product(parent_set,test_child))
            s1 = st.Structure(l1, indxs1, vals1, eds1, tot_vars_count)
            g1 = ng.NetworkGraph(s1)
            #g1.init_graph()
            g1.fast_init(test_child)
            p1 = pe.ParametersEstimator(self.sample_path, g1)
            #p1.init_sets_cims_container()
            p1.fast_init(test_child)
            sofc1 = p1.compute_parameters_for_node(test_child)
            
            #sofc1 = p1.sets_of_cims_struct.sets_of_cims[g1.get_positional_node_indx(test_child)]
            if not p_set:
                self.cache.put(test_child, sofc1)
            else:
                self.cache.put(set(p_set), sofc1)
        sofc2 = None
        #p_set.append(test_parent)
        p_set.insert(0, test_parent)
        if p_set:
            #print("FULL PSET TRYING TO FIND", p_set)
            #p_set.append(test_parent)
            #print("PSET ", p_set)
            #set_p_set = set(p_set)
            sofc2 = self.cache.find(set(p_set))
            #if sofc2:
                #print("Sofc2 in CACHE ", sofc2.actual_cims)
            #print(self.cache.list_of_sets_of_indxs)

        """p2 = pe.ParametersEstimator(self.sample_path, g2)
        p2.init_sets_cims_container()
        #p2.compute_parameters()
        p2.compute_parameters_for_node(test_child)
        sofc2 = p2.sets_of_cims_struct.sets_of_cims[s2.get_positional_node_indx(test_child)]"""
        if not sofc2:
            #print("Cache MISSS SOFC2")
            complete_info.append(test_parent)
            bool_mask2 = np.isin(self.nodes, complete_info)
            #print("BOOL MASK 2",bool_mask2)
            l2 = list(self.nodes[bool_mask2])
            #print("L2", l2)
            indxs2 = self.nodes_indxs[bool_mask2]
            #print("INDXS 2", indxs2)
            vals2 = self.nodes_vals[bool_mask2]
            eds2 = list(itertools.product(p_set, test_child))
            s2 = st.Structure(l2, indxs2, vals2, eds2, tot_vars_count)
            g2 = ng.NetworkGraph(s2)
            #g2.init_graph()
            g2.fast_init(test_child)
            p2 = pe.ParametersEstimator(self.sample_path, g2)
            #p2.init_sets_cims_container()
            p2.fast_init(test_child)
            sofc2 = p2.compute_parameters_for_node(test_child)
            #sofc2 = p2.sets_of_cims_struct.sets_of_cims[g2.get_positional_node_indx(test_child)]
            #if p_set:
                #set_p_set = set(p_set)
            self.cache.put(set(p_set), sofc2)
        for cim1, p_comb in zip(sofc1.actual_cims, sofc1.p_combs):
            #print("GETTING THIS P COMB", p_comb)
            #if len(parent_set) > 1:
            cond_cims = sofc2.filter_cims_with_mask(cims_filter, p_comb)
            #else:
                #cond_cims = sofc2.actual_cims
            #print("COnd Cims", cond_cims)
            for cim2 in cond_cims:
                #cim2 = sofc2.actual_cims[j]
                #print(indx)
                #print("Run Test", i, j)
                if not self.independence_test(child_states_numb, cim1, cim2):
                    return False
        return True

    def independence_test(self, child_states_numb, cim1, cim2):
        M1 = cim1.state_transition_matrix
        M2 = cim2.state_transition_matrix
        r1s = M1.diagonal()
        r2s = M2.diagonal()
        C1 = cim1.cim
        C2 = cim2.cim
        #print("C1", C1)
        #print("C2", C2)
        F_stats = C2.diagonal() / C1.diagonal()
        exp_alfa = self.exp_test_sign
        for val in range(0, child_states_numb):
            if F_stats[val] < f_dist.ppf(exp_alfa / 2, r1s[val], r2s[val]) or \
                    F_stats[val] > f_dist.ppf(1 - exp_alfa / 2, r1s[val], r2s[val]):
                #print("CONDITIONALLY DEPENDENT EXP")
                return False
        #M1_no_diag = self.remove_diagonal_elements(cim1.state_transition_matrix)
        #M2_no_diag = self.remove_diagonal_elements(cim2.state_transition_matrix)
        M1_no_diag = M1[~np.eye(M1.shape[0], dtype=bool)].reshape(M1.shape[0], -1)
        M2_no_diag = M2[~np.eye(M2.shape[0], dtype=bool)].reshape(
            M2.shape[0], -1)
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
                #print("CONDITIONALLY DEPENDENT CHI")
                return False
            #print("Chi test", Chi)
        return True

    def one_iteration_of_CTPC_algorithm(self, var_id, tot_vars_count):
        print("##################TESTING VAR################", var_id)
        u = list(self.complete_graph.predecessors(var_id))
        #tests_parents_numb = len(u)
        #complete_frame = self.complete_graph_frame
        #test_frame = complete_frame.loc[complete_frame['To'].isin([var_id])]
        child_states_numb = self.sample_path.structure.get_states_number(var_id)
        b = 0
        while b < len(u):
            #for parent_id in u:
            parent_indx = 0
            while parent_indx < len(u):
                #print("Parent_indx",parent_indx)
                #print("LEN U", len(u))

                removed = False
                #if not list(self.generate_possible_sub_sets_of_size(u, b, u[parent_indx])):
                    #break
                S = self.generate_possible_sub_sets_of_size(u, b, u[parent_indx])
                #print("U Set", u)
                #print("S", S)
                test_parent = u[parent_indx]
                #print("Test Parent", test_parent)
                for parents_set in S:
                    #print("Parent Set", parents_set)
                    #print("Test Parent", test_parent)
                    if self.complete_test(test_parent, var_id, parents_set, child_states_numb, tot_vars_count):
                        #print("Removing EDGE:", test_parent, var_id)
                        self.complete_graph.remove_edge(test_parent, var_id)
                        u.remove(test_parent)
                        removed = True
                        break
                    #else:
                        #parent_indx += 1
                if not removed:
                    parent_indx += 1
            b += 1
        self.cache.clear()

    def generate_possible_sub_sets_of_size(self, u, size, parent_indx):
        list_without_test_parent = u[:]
        list_without_test_parent.remove(parent_indx)
        return map(list, itertools.combinations(list_without_test_parent, size))

    def remove_diagonal_elements(self, matrix):
        m = matrix.shape[0]
        strided = np.lib.stride_tricks.as_strided
        s0, s1 = matrix.strides
        return strided(matrix.ravel()[1:], shape=(m - 1, m), strides=(s0 + s1, s1)).reshape(m, -1)

    def ctpc_algorithm(self):
        ctpc_algo = self.one_iteration_of_CTPC_algorithm
        total_vars_numb = self.sample_path.total_variables_count
        [ctpc_algo(n, total_vars_numb) for n in self.nodes]

