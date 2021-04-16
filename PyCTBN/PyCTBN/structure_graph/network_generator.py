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

    def generate_graph(self, density):
        edges = [(i, j) for i in self._labels for j in self._labels if np.random.binomial(1, density) == 1 and i != j]
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

            if len(p_info[0]) != 0:
                node_cims = []
                for comb in combs:
                    cim = self.__generate_cim(min_val, max_val, self._vals[i])
                    node_cims.append(ConditionalIntensityMatrix(cim = cim))
            else:
                node_cims = []
                cim = self.__generate_cim(min_val, max_val, self._vals[i])
                node_cims.append(ConditionalIntensityMatrix(cim = cim))
                
            self._cims[l] = SetOfCims(node_id = l, parents_states_number = p_info[2], node_states_number = self._vals[i], p_combs = combs, 
                cims = np.array(node_cims))

    def __generate_cim(self, min_val, max_val, shape):
        cim = np.empty(shape=(shape, shape))
        cim[:] = np.nan
        
        for i, c in enumerate(cim):
            diag = (max_val - min_val) * np.random.random_sample() + min_val
            row = np.random.rand(1, len(cim))[0]
            row /= (sum(row) - row[i])
            row *= diag
            row[i] = -1 * diag
            cim[i] = row

        return cim

    def out_json(self, filename):
        dyn_str = [{"From": edge[0], "To": edge[1]} for edge in self._graph.edges]
        variables = [{"Name": l, "Value": self._vals[i]} for i, l in enumerate(self._labels)]
        dyn_cims = {}

        for i, l in enumerate(self._labels):
            dyn_cims[l] = {}
            parents = self._graph.get_ordered_by_indx_set_of_parents(l)[0]
            for j, comb in enumerate(self._cims[l].p_combs):
                comb_key = ""
                for k, val in enumerate(comb):
                    comb_key += parents[k] + "=" + str(val)
                    if k < len(comb) - 1:
                        comb_key += ","

                cim = self._cims[l].filter_cims_with_mask(np.array([True for p in parents]), comb)
                if len(parents) == 1:
                    cim = cim[comb[0]].cim
                elif len(parents) == 0:
                    cim = cim[0].cim
                else:
                    cim = cim.cim
                
                dyn_cims[l][comb_key] = [dict([(str(i), val) for i, val in enumerate(row)]) for row in cim]

        data = {
            "dyn.str": dyn_str,
            "variables": variables,
            "dyn.cims": dyn_cims,
            "samples": []
        }

        print(data)

    """ path = os.getcwd()
    with open(path + "/" + filename, "w") as json_file:
        json.dump(data, json_file) """

    @property
    def graph(self) -> NetworkGraph:
        return self._graph

    @property
    def cims(self) -> NetworkGraph:
        return self._cims   