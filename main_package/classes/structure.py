

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

    def list_of_edges(self):
        edges_list = []
        for indx, row in self.structure_frame.iterrows():
            row_tuple = (row.From, row.To)
            edges_list.append(row_tuple)
        return edges_list

    def list_of_nodes(self):
        return self.variables_frame['Name'] #TODO rimuovere dipendenza diretta dalla key 'Name'

    def get_node_id(self, node_indx):
        return self.variables_frame['Name'][node_indx]

    def get_node_indx(self, node_id):
        return list(self.variables_frame['Name']).index(node_id)

    def get_states_number(self):
        return self.variables_frame['Value'][0]
