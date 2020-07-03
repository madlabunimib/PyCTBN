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

    def __init__(self, node_id, parents_states_number, node_states_number):
        self.node_id = node_id
        self.parents_states_number = parents_states_number
        self.node_states_number = node_states_number
        self.actual_cims = None
        self.build_actual_cims_structure()

    def build_actual_cims_structure(self):
        cims_number = 1
        for state_number in self.parents_states_number:
            cims_number = cims_number * state_number
        self.actual_cims = np.empty(cims_number, dtype=cim.ConditionalIntensityMatrix)
        for indx, matrix in enumerate(self.actual_cims):
            self.actual_cims[indx] = cim.ConditionalIntensityMatrix(self.node_states_number)

    def update_state_transition(self, indexes, element_indx_tuple):
        matrix_indx = self.indexes_converter(indexes)
        self.actual_cims[matrix_indx].update_state_transition_count(element_indx_tuple)

    def update_state_residence_time(self, which_matrix, which_element, time):
        matrix_indx = self.indexes_converter(which_matrix)
        self.actual_cims[matrix_indx].update_state_residence_time_for_state(which_element, time)


    def get_cims_number(self):
        return len(self.actual_cims)

    def indexes_converter(self, indexes): # Si aspetta array del tipo [2,2] dove
        #print(type(indexes))
        if indexes.size == 0:
            return 0
        else:
            vector_index = 0
            for indx, value in enumerate(indexes):
                vector_index = vector_index*self.parents_states_number[indx] + indexes[indx]
            return vector_index


"""
sofc = SetOfCims('W', [], 2)
sofc.build_actual_cims_structure()
print(sofc.actual_cims)
print(sofc.indexes_converter([]))"""



