
import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist
from tqdm import tqdm

from .cache import Cache
from .conditional_intensity_matrix import ConditionalIntensityMatrix
from .network_graph import NetworkGraph
from .parameters_estimator import ParametersEstimator
from .sample_path import SamplePath
from .structure import Structure


class StructureEstimator(object):
    """Has the task of estimating the network structure given the trajectories in ``samplepath``.

    :param sample_path: the _sample_path object containing the trajectories and the real structure
    :type sample_path: SamplePath
    :param exp_test_alfa: the significance level for the exponential Hp test
    :type exp_test_alfa: float
    :param chi_test_alfa: the significance level for the chi Hp test
    :type chi_test_alfa: float
    :_nodes: the nodes labels
    :_nodes_vals: the nodes cardinalities
    :_nodes_indxs: the nodes indexes
    :_complete_graph: the complete directed graph built using the nodes labels in ``_nodes``
    :_cache: the Cache object
    """

    def __init__(self, sample_path: SamplePath, exp_test_alfa: float, chi_test_alfa: float):
        """Constructor Method
        """
        self._sample_path = sample_path
        self._nodes = np.array(self._sample_path.structure.nodes_labels)
        self._nodes_vals = self._sample_path.structure.nodes_values
        self._nodes_indxs = self._sample_path.structure.nodes_indexes
        self._complete_graph = StructureEstimator.build_complete_graph(self._sample_path.structure.nodes_labels)
        self._exp_test_sign = exp_test_alfa
        self._chi_test_alfa = chi_test_alfa
        self._cache = Cache()

    @staticmethod
    def build_complete_graph(node_ids: typing.List) -> nx.DiGraph:
        """Builds a complete directed graph (no self loops) given the nodes labels in the list ``node_ids``:

        :param node_ids: the list of nodes labels
        :type node_ids: List
        :return: a complete Digraph Object
        :rtype: networkx.DiGraph
        """
        complete_graph = nx.DiGraph()
        complete_graph.add_nodes_from(node_ids)
        complete_graph.add_edges_from(itertools.permutations(node_ids, 2))
        return complete_graph

    def complete_test(self, test_parent: str, test_child: str, parent_set: typing.List, child_states_numb: int,
                      tot_vars_count: int) -> bool:
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
            eds2 = list(itertools.product(p_set, test_child))
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

        for cim1, p_comb in zip(sofc1.actual_cims, sofc1.p_combs):
            cond_cims = sofc2.filter_cims_with_mask(cims_filter, p_comb)
            for cim2 in cond_cims:
                if not self.independence_test(child_states_numb, cim1, cim2):
                    return False
        return True

    def independence_test(self, child_states_numb: int, cim1: ConditionalIntensityMatrix,
                          cim2: ConditionalIntensityMatrix) -> bool:
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

    def one_iteration_of_CTPC_algorithm(self, var_id: str, tot_vars_count: int) -> None:
        """Performs an iteration of the CTPC algorithm using the node ``var_id`` as ``test_child``.

        :param var_id: the node label of the test child
        :type var_id: string
        :param tot_vars_count: the number of _nodes in the net
        :type tot_vars_count: int
        """
        u = list(self._complete_graph.predecessors(var_id))
        child_states_numb = self._sample_path.structure.get_states_number(var_id)
        b = 0
        while b < len(u):
            parent_indx = 0
            while parent_indx < len(u):
                removed = False
                S = StructureEstimator.generate_possible_sub_sets_of_size(u, b, u[parent_indx])
                test_parent = u[parent_indx]
                for parents_set in S:
                    if self.complete_test(test_parent, var_id, parents_set, child_states_numb, tot_vars_count):
                        self._complete_graph.remove_edge(test_parent, var_id)
                        u.remove(test_parent)
                        removed = True
                        break
                if not removed:
                    parent_indx += 1
            b += 1
        self._cache.clear()

    @staticmethod
    def generate_possible_sub_sets_of_size(u: typing.List, size: int, parent_label: str) -> \
            typing.Iterator:
        """Creates a list containing all possible subsets of the list ``u`` of size ``size``,
        that do not contains a the node identified by ``parent_label``.

        :param u: the list of nodes
        :type u: List
        :param size: the size of the subsets
        :type size: int
        :param parent_label: the node to exclude in the subsets generation
        :type parent_label: string
        :return: an Iterator Object containing a list of lists
        :rtype: Iterator
        """
        list_without_test_parent = u[:]
        list_without_test_parent.remove(parent_label)
        return map(list, itertools.combinations(list_without_test_parent, size))

    def ctpc_algorithm(self) -> None:
        """Compute the CTPC algorithm over the entire net.
        """
        ctpc_algo = self.one_iteration_of_CTPC_algorithm
        total_vars_numb = self._sample_path.total_variables_count
        [ctpc_algo(n, total_vars_numb) for n in tqdm(self._nodes)]

    def save_results(self) -> None:
        """Save the estimated Structure to a .json file in the path where the data are loaded from.
        The file is named as the input dataset but the `results_` word is appended to the results file.
        """
        res = json_graph.node_link_data(self._complete_graph)
        name = self._sample_path._importer.file_path.rsplit('/', 1)[-1]
        name = name.split('.', 1)[0]
        name += '_' + str(self._sample_path._importer.dataset_id())
        name += '.json'
        file_name = 'results_' + name
        with open(file_name, 'w') as f:
            json.dump(res, f)

    def adjacency_matrix(self) -> np.ndarray:
        """Converts the estimated structrure ``_complete_graph`` to a boolean adjacency matrix representation.

        :return: The adjacency matrix of the graph ``_complete_graph``
        :rtype: numpy.ndArray
        """
        return nx.adj_matrix(self._complete_graph).toarray().astype(bool)


