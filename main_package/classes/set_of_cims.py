import numpy as np
import conditional_intensity_matrix as cim


class SetOfCims:

    def __init__(self, node_id, ordered_parent_set, value_type):
        self.node_id = node_id
        self.ordered_parent_set = ordered_parent_set
        self.value = value_type
        self.actual_cims = None

    def build_actual_cims_structure(self):
        cims_number = self.value**len(self.ordered_parent_set)
        self.actual_cims = np.empty(cims_number, dtype=cim.ConditionalIntensityMatrix)
        for indx, matrix in enumerate(self.actual_cims):
            self.actual_cims[indx] = cim.ConditionalIntensityMatrix(self.value)

    def indexes_converter(self, dict_of_indexes): # Si aspetta oggetti del tipo {X:1, Y:1, Z:0}
        literal_index = ""
        for node in self.ordered_parent_set:
            literal_index = literal_index + str(dict_of_indexes[node])
        return int(literal_index, self.value)



sofc = SetOfCims('W', ['X','Y', 'Z'], 2)
#sofc.build_actual_cims_structure(sofc.ordered_parent_set, sofc.value)

sofc.build_actual_cims_structure()
print(sofc.actual_cims)
print(sofc.indexes_converter({'X':1, 'Y':1, 'Z':0}))



