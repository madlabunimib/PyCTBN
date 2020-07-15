

class Structure:
    """
    Contiene tutte il informazioni sulla struttura della rete (connessione dei nodi, valori assumibili dalle variabili)

    :structure_frame: il dataframe contenente le connessioni dei nodi della rete
    :variables_frame: il data_frame contenente i valori assumibili dalle variabili e si suppone il corretto ordinamento
    rispetto alle colonne del dataset
    """

    def __init__(self, structure, variables):
        self.structure_frame = structure
        self.variables_frame = variables
        self.name_label = variables.columns.values[0]
        self.value_label = variables.columns.values[1]

    def list_of_edges(self):
        #edges_list = []
        #for indx, row in self.structure_frame.iterrows():
            #row_tuple = (row[0], row[1])
            #edges_list.append(row_tuple)
        records = self.structure_frame.to_records(index=False)
        edges_list = list(records)
        return edges_list

    def list_of_nodes_labels(self):
        return self.variables_frame[self.name_label].values.tolist()

    def list_of_nodes_indexes(self):
        nodes_indexes = []
        for indx in self.list_of_nodes_labels():
            nodes_indexes.append(indx)
        return nodes_indexes

    def get_node_id(self, node_indx):
        return self.variables_frame[self.name_label][node_indx]

    def get_node_indx(self, node_id):
        return list(self.variables_frame[self.name_label]).index(node_id)

    def get_states_number(self, node):
        return self.variables_frame[self.value_label][self.get_node_indx(node)]

    def get_states_number_by_indx(self, node_indx):
        #print(self.value_label)
        return self.variables_frame[self.value_label][node_indx]

    def __repr__(self):
        return "Variables:\n" + str(self.variables_frame) + "\nEdges: \n" + str(self.structure_frame)
