

class Structure:

    def __init__(self, structure, variables):
        self.structure_frame = structure
        self.variables_frame = variables

    def list_of_edges(self):
        edges_list = []
        for indx, row in self.structure_frame.iterrows():
           row_tuple = (row.From, row.To)
           edges_list.append(row_tuple)
        return edges_list


