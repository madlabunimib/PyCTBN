import numpy as np
import conditional_intensity_matrix as cim


class SetOfCims:
    """
    Rappresenta la struttura che aggrega tutte le CIM per la variabile di label node_id

    :node_id: la label della varibile a cui fanno riferimento le CIM
    :ordered_parent_set: il set dei parent della variabile node_id ordinata secondo la property indx
    :value: il numero massimo di stati assumibili dalla variabile
    :actual_cims: le CIM della varibile
    """

    def __init__(self, node_id, ordered_parent_set, value_type):
        self.node_id = node_id
        self.ordered_parent_set = ordered_parent_set
        self.value = value_type
        self.actual_cims = None
        self.build_actual_cims_structure()

    def build_actual_cims_structure(self):
        cims_number = self.value**len(self.ordered_parent_set)
        self.actual_cims = np.empty(cims_number, dtype=cim.ConditionalIntensityMatrix)
        for indx, matrix in enumerate(self.actual_cims):
            self.actual_cims[indx] = cim.ConditionalIntensityMatrix(self.value)

    def update_state_transition(self, dict_of_indexes, element_indx_tuple):
        matrix_indx = self.indexes_converter(dict_of_indexes)
        self.actual_cims[matrix_indx].update_state_transition_count(element_indx_tuple)

    def update_state_residence_time(self, which_matrix, which_element, time):
        matrix_indx = self.indexes_converter(which_matrix)
        self.actual_cims[matrix_indx].update_state_residence_time_for_state(which_element, time)


    def get_cims_number(self):
        return len(self.actual_cims)

    def indexes_converter(self, dict_of_indexes): # Si aspetta oggetti del tipo {X:1, Y:1, Z:0} dove
                                                    # le keys sono i parent del nodo e i values sono i valori che assumono
        if not dict_of_indexes:
            return 0
        else:
            literal_index = ""
            for node in self.ordered_parent_set:
                literal_index = literal_index + str(dict_of_indexes[node])
                #print(literal_index)
            return int(literal_index, self.value)



"""sofc = SetOfCims('W', ['X','Y', 'Z'], 2)
sofc.build_actual_cims_structure()
print(sofc.actual_cims)
print(sofc.indexes_converter({'X':1, 'Y':1, 'Z':0}))"""



