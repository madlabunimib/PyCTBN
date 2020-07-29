import numpy as np


class Structure:
    """
    Contiene tutte il informazioni sulla struttura della rete (connessione dei nodi, valori assumibili dalle variabili)

    :structure_frame: il dataframe contenente le connessioni dei nodi della rete
    :variables_frame: il data_frame contenente i valori assumibili dalle variabili e si suppone il corretto ordinamento
    rispetto alle colonne del dataset
    """

    def __init__(self, nodes_label_list, node_indexes_arr, nodes_vals_arr, edges_list, total_variables_number):
        #self.structure_frame = structure
        #self.variables_frame = variables
        self.nodes_labels_list = nodes_label_list
        self.nodes_indexes_arr = node_indexes_arr
        self.nodes_vals_arr = nodes_vals_arr
        self.edges_list = edges_list
        self.total_variables_number = total_variables_number
        #self.name_label = variables.columns.values[0]
        #self.value_label = variables.columns.values[1]

    def list_of_edges(self):
        #records = self.structure_frame.to_records(index=False)
        #edges_list = list(records)
        return self.edges_list

    def list_of_nodes_labels(self):
        return self.nodes_labels_list

    def list_of_nodes_indexes(self):
        return self.nodes_indexes_arr

    def get_node_id(self, node_indx):
        return self.nodes_labels_list[node_indx]

    def get_node_indx(self, node_id):
        return self.nodes_indexes_arr[self.nodes_labels_list.index(node_id)]

    def get_positional_node_indx(self, node_id):
        return self.nodes_labels_list.index(node_id)

    def get_states_number(self, node):
        #print("node", node)
        return self.nodes_vals_arr[self.get_positional_node_indx(node)]

    def get_states_number_by_indx(self, node_indx):
        #print(self.value_label)
        #print("Node indx", node_indx)
        return self.nodes_vals_arr[node_indx]

    def nodes_values(self):
        return self.nodes_vals_arr

    def total_variables_number(self):
        return self.total_variables_number

    def __repr__(self):
        return "Variables:\n" + str(self.variables_frame) + "\nEdges: \n" + str(self.structure_frame)

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Structure):
            return self.structure_frame.equals(other.structure_frame) and \
                   self.variables_frame.equals(other.variables_frame)
        return NotImplemented

    """def remove_node(self, node_id):
        self.variables_frame = self.variables_frame[self.variables_frame.Name != node_id]
        self.structure_frame = self.structure_frame[(self.structure_frame.From != node_id) &
                                                    (self.structure_frame.To != node_id)]"""
