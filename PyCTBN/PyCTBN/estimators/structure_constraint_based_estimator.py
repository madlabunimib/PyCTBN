
import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import os
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist
from tqdm import tqdm

from ..utility.cache import Cache
from ..structure_graph.conditional_intensity_matrix import ConditionalIntensityMatrix
from ..structure_graph.network_graph import NetworkGraph
from .parameters_estimator import ParametersEstimator
from .structure_estimator import StructureEstimator
from ..structure_graph.sample_path import SamplePath
from ..structure_graph.structure import Structure
from ..optimizers.constraint_based_optimizer import ConstraintBasedOptimizer

import concurrent.futures



import multiprocessing
from multiprocessing import Pool


class StructureConstraintBasedEstimator(StructureEstimator):
    """
    Has the task of estimating the network structure given the trajectories in samplepath by using a constraint-based approach.

    :param sample_path: the _sample_path object containing the trajectories and the real structure
    :type sample_path: SamplePath
    :param exp_test_alfa: the significance level for the exponential Hp test
    :type exp_test_alfa: float
    :param chi_test_alfa: the significance level for the chi Hp test
    :type chi_test_alfa: float
    :param known_edges: the prior known edges in the net structure if present
    :type known_edges: List
    :param thumb_threshold: the threshold value to consider a valid independence test
    :type thumb_threshold: int
    :_nodes: the nodes labels
    :_nodes_vals: the nodes cardinalities
    :_nodes_indxs: the nodes indexes
    :_complete_graph: the complete directed graph built using the nodes labels in ``_nodes``
    :_cache: the Cache object
    """

    def __init__(self, sample_path: SamplePath, exp_test_alfa: float, chi_test_alfa: float,known_edges: typing.List= [],
                 thumb_threshold:int = 25):
        super().__init__(sample_path,known_edges)
        self._exp_test_sign = exp_test_alfa
        self._chi_test_alfa = chi_test_alfa
        self._thumb_threshold = thumb_threshold
        self._cache = Cache()

    def complete_test(self, test_parent: str, test_child: str, parent_set: typing.List, child_states_numb: int,
                      tot_vars_count: int, parent_indx, child_indx) -> bool:
        """Performs a complete independence test on the directed graphs G1 = {test_child U parent_set}
        G2 = {G1 U test_parent} (added as an additional parent of the test_child).
        Generates all the necessary structures and datas to perform the tests.

        :param test_parent: the node label of the test parent
        :type test_parent: string
        :param test_child: the node label of the child
        :type test_child: string
        :param parent_set: the common parent set
        :type parent_set: List
        :param child_states_numb: the cardinality of the ``test_child``
        :type child_states_numb: int
        :param tot_vars_count: the total number of variables in the net
        :type tot_vars_count: int
        :return: True iff test_child and test_parent are independent given the sep_set parent_set. False otherwise
        :rtype: bool
        """
        p_set = parent_set[:]
        complete_info = parent_set[:]
        complete_info.append(test_child)

        parents = np.array(parent_set)
        parents = np.append(parents, test_parent)
        sorted_parents = self._nodes[np.isin(self._nodes, parents)]
        cims_filter = sorted_parents != test_parent

        p_set.insert(0, test_parent)
        sofc2 = self._cache.find(set(p_set))

        if not sofc2:
            complete_info.append(test_parent)
            bool_mask2 = np.isin(self._nodes, complete_info)
            l2 = list(self._nodes[bool_mask2])
            indxs2 = self._nodes_indxs[bool_mask2]
            vals2 = self._nodes_vals[bool_mask2]
            eds2 = list(itertools.product(p_set, [test_child]))
            s2 = Structure(l2, indxs2, vals2, eds2, tot_vars_count)
            g2 = NetworkGraph(s2)
            g2.fast_init(test_child)
            p2 = ParametersEstimator(self._sample_path.trajectories, g2)
            p2.fast_init(test_child)
            sofc2 = p2.compute_parameters_for_node(test_child)
            self._cache.put(set(p_set), sofc2)

        del p_set[0]
        sofc1 = self._cache.find(set(p_set))
        if not sofc1:
            g2.remove_node(test_parent)
            g2.fast_init(test_child)
            p2 = ParametersEstimator(self._sample_path.trajectories, g2)
            p2.fast_init(test_child)
            sofc1 = p2.compute_parameters_for_node(test_child)
            self._cache.put(set(p_set), sofc1)
        thumb_value = 0.0
        if child_states_numb > 2:
            parent_val = self._sample_path.structure.get_states_number(test_parent)
            bool_mask_vals = np.isin(self._nodes, parent_set)
            parents_vals = self._nodes_vals[bool_mask_vals]
            thumb_value = self.compute_thumb_value(parent_val, child_states_numb, parents_vals)
        for cim1, p_comb in zip(sofc1.actual_cims, sofc1.p_combs):
            cond_cims = sofc2.filter_cims_with_mask(cims_filter, p_comb)
            for cim2 in cond_cims:
                if not self.independence_test(child_states_numb, cim1, cim2, thumb_value, parent_indx, child_indx):
                    return False
        return True

    def independence_test(self, child_states_numb: int, cim1: ConditionalIntensityMatrix,
                          cim2: ConditionalIntensityMatrix, thumb_value: float, parent_indx, child_indx) -> bool:
        """Compute the actual independence test using two cims.
        It is performed first the exponential test and if the null hypothesis is not rejected,
        it is performed also the chi_test.

        :param child_states_numb: the cardinality of the test child
        :type child_states_numb: int
        :param cim1: a cim belonging to the graph without test parent
        :type cim1: ConditionalIntensityMatrix
        :param cim2: a cim belonging to the graph with test parent
        :type cim2: ConditionalIntensityMatrix
        :return: True iff both tests do NOT reject the null hypothesis of independence. False otherwise.
        :rtype: bool
        """
        M1 = cim1.state_transition_matrix
        M2 = cim2.state_transition_matrix
        r1s = M1.diagonal()
        r2s = M2.diagonal()
        C1 = cim1.cim
        C2 = cim2.cim
        if child_states_numb > 2 and (np.sum(np.diagonal(M1)) / thumb_value) < self._thumb_threshold:
                self._removable_edges_matrix[parent_indx][child_indx] = False
                return False
        F_stats = C2.diagonal() / C1.diagonal()
        exp_alfa = self._exp_test_sign
        for val in range(0, child_states_numb):
            if F_stats[val] < f_dist.ppf(exp_alfa / 2, r1s[val], r2s[val]) or \
                    F_stats[val] > f_dist.ppf(1 - exp_alfa / 2, r1s[val], r2s[val]):
                return False
        M1_no_diag = M1[~np.eye(M1.shape[0], dtype=bool)].reshape(M1.shape[0], -1)
        M2_no_diag = M2[~np.eye(M2.shape[0], dtype=bool)].reshape(
            M2.shape[0], -1)
        chi_2_quantile = chi2_dist.ppf(1 - self._chi_test_alfa, child_states_numb - 1)
        Ks = np.sqrt(r1s / r2s)
        Ls = np.sqrt(r2s / r1s)
        for val in range(0, child_states_numb):
            Chi = np.sum(np.power(Ks[val] * M2_no_diag[val] - Ls[val] *M1_no_diag[val], 2) /
                         (M1_no_diag[val] + M2_no_diag[val]))
            if Chi > chi_2_quantile:
                return False
        return True
        
    def compute_thumb_value(self, parent_val, child_val, parent_set_vals):
        """Compute the value to test against the thumb_threshold.
        
        :param parent_val: test parent's variable cardinality
        :type parent_val: int
        :param child_val: test child's variable cardinality
        :type child_val: int
        :param parent_set_vals: the cardinalities of the nodes in the current sep-set
        :type parent_set_vals: List
        :return: the thumb value for the current independence test
        :rtype: int
        """
        df = (child_val - 1) ** 2
        df = df * parent_val
        for v in parent_set_vals:
            df = df * v
        return df
        
    def one_iteration_of_CTPC_algorithm(self, var_id: str, tot_vars_count: int)-> typing.List:
        """Performs an iteration of the CTPC algorithm using the node ``var_id`` as ``test_child``.

        :param var_id: the node label of the test child
        :type var_id: string
        """
        optimizer_obj = ConstraintBasedOptimizer(
                                                            node_id = var_id,
                                                            structure_estimator = self,
                                                            tot_vars_count = tot_vars_count)
        return optimizer_obj.optimize_structure()

    
    def ctpc_algorithm(self, disable_multiprocessing:bool= False, processes_number:int= None):
        """Compute the CTPC algorithm over the entire net.

        :param disable_multiprocessing: true if you desire to disable the multiprocessing operations, default to False
        :type disable_multiprocessing: Boolean, optional
        :param processes_number: if disable_multiprocessing is false indicates 
        the maximum number of process; if None it will be automatically set, default to None
        :type processes_number: int, optional
        """
        ctpc_algo = self.one_iteration_of_CTPC_algorithm
        total_vars_numb = self._sample_path.total_variables_count

        n_nodes= len(self._nodes)

        total_vars_numb_array =  [total_vars_numb] *  n_nodes

        'get the number of CPU'
        cpu_count = multiprocessing.cpu_count()

        'Remove all the edges from the structure'   
        self._sample_path.structure.clean_structure_edges()

        'Estimate the best parents for each node'
        #with multiprocessing.Pool(processes=cpu_count) as pool:
        #with get_context("spawn").Pool(processes=cpu_count) as pool:
        if disable_multiprocessing:
            print("DISABLED")
            cpu_count = 1
            list_edges_partial = [ctpc_algo(n,total_vars_numb) for n in self._nodes]
        else:
            if processes_number is not None and cpu_count > processes_number:
                cpu_count = processes_number

            print(f"CPU COUNT: {cpu_count}")
            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                list_edges_partial = executor.map(ctpc_algo,
                                                                 self._nodes,
                                                                 total_vars_numb_array)
            
        'Update the graph'
        edges = set(itertools.chain.from_iterable(list_edges_partial))
        self._complete_graph = nx.DiGraph()
        self._complete_graph.add_edges_from(edges)

        return edges

        
    def estimate_structure(self, disable_multiprocessing:bool=False, processes_number:int= None):
        """
        Compute the constraint-based algorithm to find the optimal structure

        :param disable_multiprocessing: true if you desire to disable the multiprocessing operations, default to False
        :type disable_multiprocessing: Boolean, optional
        :param processes_number: if disable_multiprocessing is false indicates 
        the maximum number of process; if None it will be automatically set, default to None
        :type processes_number: int, optional
        """
        return self.ctpc_algorithm(disable_multiprocessing=disable_multiprocessing,
                                    processes_number=processes_number)

    


