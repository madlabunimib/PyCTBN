

class Structure:

    def __init__(self, structure, variables):
        self.structure_frame = structure
        self.variables_frame = variables

    def list_of_edge_ids(self):
        edges_list = []
        for indx, row in self.structure_frame.iterrows():
           row_tuple = (row.From, row.To)
           edges_list.append(row_tuple)
        return edges_list

    def list_of_node_ids(self):
        #print(self.variables_frame)
        return self.variables_frame['Name']

    def get_node_id(self, node_indx):
        return self.variables_frame['Name'][node_indx]

    def get_node_indx(self, node_id):
        return list(self.variables_frame['Name']).index(node_id)
