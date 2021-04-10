from .structure import Structure
from .network_graph import NetworkGraph
from .conditional_intensity_matrix import ConditionalIntensityMatrix
from .set_of_cims import SetOfCims
import numpy as np

class NetworkGenerator(object):
    def __init__(self, labels, vals):
        self._labels = labels
        self._vals = vals
        self._indxs = np.array([i for i, l in enumerate(labels)])
        self._graph = None
        self._cims = None

    def generate_graph(self):
        edges = [(i, j) for i in self._labels for j in self._labels if np.random.binomial(1, 0.5) == 1 and i != j]
        indxs = np.array([i for i, l in enumerate(self._labels)])
        s = Structure(self._labels, self._indxs, self._vals, edges, len(self._labels))
        self._graph = NetworkGraph(s)
        self._graph.add_nodes(s.nodes_labels)
        self._graph.add_edges(s.edges)

    def generate_cims(self, min_val, max_val):
        if self._graph is None:
            return

        self._cims = {}

        for i, l in enumerate(self._labels):
            p_info = self._graph.get_ordered_by_indx_set_of_parents(l)
            combs = self._graph.build_p_comb_structure_for_a_node(p_info[2])

            node_cims = []
            for comb in combs:
                cim = np.empty(shape=(self._vals[i], self._vals[i]))
                cim[:] = np.nan
                
                for i, c in enumerate(cim):
                    diag = (max_val - min_val) * np.random.random_sample() + min_val
                    row = np.random.rand(1, len(cim))[0]
                    row /= (sum(row) - row[i])
                    row *= diag
                    row[i] = -1 * diag
                    cim[i] = row

                node_cims.append(ConditionalIntensityMatrix(cim = cim))
                
            self._cims[l] = SetOfCims(node_id = l, parents_states_number = p_info[2], node_states_number = self._vals[i], p_combs = combs, 
                cims = np.array(node_cims))

    @property
    def graph(self) -> NetworkGraph:
        return self._graph

    @property
    def cims(self) -> NetworkGraph:
        return self._cims   