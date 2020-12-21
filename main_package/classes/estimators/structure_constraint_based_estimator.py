import sys
sys.path.append('../')
import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist

import utility.cache as ch
import structure_graph.conditional_intensity_matrix as condim
import structure_graph.network_graph as ng
import estimators.parameters_estimator as pe
import estimators.structure_estimator as se
import structure_graph.sample_path as sp
import structure_graph.structure as st
import optimizers.constraint_based_optimizer as optimizer

from utility.decorators import timing,timing_write

import multiprocessing
from multiprocessing import Pool

from multiprocessing import get_context

class StructureConstraintBasedEstimator(se.StructureEstimator):
    """
    Has the task of estimating the network structure given the trajectories in samplepath.

    :exp_test_sign: the significance level for the exponential Hp test
    :chi_test_alfa: the significance level for the chi Hp test
    """

    def __init__(self, sample_path: sp.SamplePath, exp_test_alfa: float, chi_test_alfa: float):
        super().__init__(sample_path)
        self.exp_test_sign = exp_test_alfa
        self.chi_test_alfa = chi_test_alfa


    def complete_test(self, test_parent: str, test_child: str, parent_set: typing.List, child_states_numb: int,
                      tot_vars_count: int):
        """
        Permorms a complete independence test on the directed graphs G1 = test_child U parent_set
        G2 = G1 U test_parent (added as an additional parent of the test_child).
        Generates all the necessary structures and datas to perform the tests.

        Parameters:
            test_parent: the node label of the test parent
            test_child: the node label of the child
            parent_set: the common parent set
            child_states_numb: the cardinality of the test_child
            tot_vars_count_ the total number of variables in the net
        Returns:
            True iff test_child and test_parent are independent given the sep_set parent_set
            False otherwise
        """
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
        #if not p_set:
            #print("EMPTY PSET TRYING TO FIND", test_child)
            #sofc1 = self.cache.find(test_child)
        #else:
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
            g1.fast_init(test_child)
            p1 = pe.ParametersEstimator(self.sample_path, g1)
            p1.fast_init(test_child)
            sofc1 = p1.compute_parameters_for_node(test_child)
            #if not p_set:
                #self.cache.put(test_child, sofc1)
            #else:
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
            g2.fast_init(test_child)
            p2 = pe.ParametersEstimator(self.sample_path, g2)
            p2.fast_init(test_child)
            sofc2 = p2.compute_parameters_for_node(test_child)
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

    def independence_test(self, child_states_numb: int, cim1: condim.ConditionalIntensityMatrix,
                          cim2: condim.ConditionalIntensityMatrix):
        """
        Compute the actual independence test using two cims.
        It is performed first the exponential test and if the null hypothesis is not rejected,
        it is permormed also the chi_test.

        Parameters:
            child_states_numb: the cardinality of the test child
            cim1: a cim belonging to the graph without test parent
            cim2: a cim belonging to the graph with test parent

        Returns:
            True iff both tests do NOT reject the null hypothesis of indipendence
            False otherwise
        """
        M1 = cim1.state_transition_matrix
        M2 = cim2.state_transition_matrix
        r1s = M1.diagonal()
        r2s = M2.diagonal()
        C1 = cim1.cim
        C2 = cim2.cim
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

    def one_iteration_of_CTPC_algorithm(self, var_id: str, tot_vars_count: int):
        """
        Performs an iteration of the CTPC algorithm using the node var_id as test_child.

        Parameters:
            var_id: the node label of the test child
            tot_vars_count: the number of nodes in the net
        Returns:
            void
        """
        optimizer_obj = optimizer.ConstraintBasedOptimizer(
                                                            node_id = var_id,
                                                            structure_estimator = self,
                                                            tot_vars_count = tot_vars_count)
        return optimizer_obj.optimize_structure()

    @timing
    def ctpc_algorithm(self,disable_multiprocessing:bool= False ):
        """
        Compute the CTPC algorithm.
        Parameters:
            void
        Returns:
            void
        """
        ctpc_algo = self.one_iteration_of_CTPC_algorithm
        total_vars_numb = self.sample_path.total_variables_count

        n_nodes= len(self.nodes)

        total_vars_numb_array =  [total_vars_numb] *  n_nodes

        'get the number of CPU'
        cpu_count = multiprocessing.cpu_count()

        if disable_multiprocessing:
            print("DISABILITATO")
            cpu_count = 1

        'Remove all the edges from the structure'   
        self.sample_path.structure.clean_structure_edges()

        'Estimate the best parents for each node'
        #with multiprocessing.Pool(processes=cpu_count) as pool:
        with get_context("spawn").Pool(processes=cpu_count) as pool:
        
            list_edges_partial = pool.starmap(ctpc_algo, zip(
                                                                 self.nodes,
                                                                 total_vars_numb_array))
            #list_edges_partial = [ctpc_algo(n,total_vars_numb) for n in self.nodes]

        return set(itertools.chain.from_iterable(list_edges_partial))

        
    @timing 
    def estimate_structure(self,disable_multiprocessing:bool=False):
        return self.ctpc_algorithm(disable_multiprocessing=disable_multiprocessing)

    


