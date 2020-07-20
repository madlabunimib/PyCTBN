import numpy as np
from numba import njit, int32
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
        self.actual_cims = []
        self.state_residence_times = None
        self.transition_matrices = None
        self.build_actual_cims_structure()

    def build_actual_cims_structure(self):
        #cims_number = 1
        #for state_number in self.parents_states_number:
            #cims_number = cims_number * state_number
        if not self.parents_states_number:
            #self.actual_cims = np.empty(1, dtype=cim.ConditionalIntensityMatrix)
            #self.actual_cims[0] = cim.ConditionalIntensityMatrix(self.node_states_number)
            self.state_residence_times = np.zeros((1, self.node_states_number), dtype=np.float)
            self.transition_matrices = np.zeros((1,self.node_states_number, self.node_states_number), dtype=np.int)
        else:
            #self.actual_cims = np.empty(self.parents_states_number, dtype=cim.ConditionalIntensityMatrix)
            #self.build_actual_cims(self.actual_cims)
        #for indx, matrix in enumerate(self.actual_cims):
            #self.actual_cims[indx] = cim.ConditionalIntensityMatrix(self.node_states_number)
            self.state_residence_times = \
                np.zeros((np.prod(self.parents_states_number), self.node_states_number), dtype=np.float)
            self.transition_matrices = np.zeros([np.prod(self.parents_states_number), self.node_states_number,
                                                 self.node_states_number], dtype=np.int)

    def update_state_transition(self, indexes, element_indx_tuple):
        #matrix_indx = self.indexes_converter(indexes)
        #print(indexes)
        if not indexes:
            self.actual_cims[0].update_state_transition_count(element_indx_tuple)
        else:
            self.actual_cims[indexes].update_state_transition_count(element_indx_tuple)

    def update_state_residence_time(self, which_matrix, which_element, time):
        #matrix_indx = self.indexes_converter(which_matrix)

        if not which_matrix:
            self.actual_cims[0].update_state_residence_time_for_state(which_element, time)
        else:
            #print(type(which_matrix))
            #print(self.actual_cims[(2,2)])
            self.actual_cims[which_matrix].update_state_residence_time_for_state(which_element, time)

    def build_actual_cims(self, cim_structure):
        for indx in range(len(cim_structure)):
            if cim_structure[indx] is None:
                cim_structure[indx] = cim.ConditionalIntensityMatrix(self.node_states_number)
            else:
                self.build_actual_cims(cim_structure[indx])

    def get_cims_number(self):
        return len(self.actual_cims)

    def indexes_converter(self, indexes): # Si aspetta array del tipo [2,2] dove
        assert len(indexes) == len(self.parents_states_number)
        vector_index = 0
        if not indexes:
            return vector_index
        else:
            for indx, value in enumerate(indexes):
                vector_index = vector_index*self.parents_states_number[indx] + indexes[indx]
            return vector_index

    def build_cims(self, state_res_times, transition_matrices):
        for state_res_time_vector, transition_matrix in zip(state_res_times, transition_matrices):
            #print(state_res_time_vector, transition_matrix)
            cim_to_add = cim.ConditionalIntensityMatrix(self.node_states_number,
                                                        state_res_time_vector, transition_matrix)
            cim_to_add.compute_cim_coefficients()
            #print(cim_to_add)
            self.actual_cims.append(cim_to_add)
        self.transition_matrices = None
        self.state_residence_times = None

    def get_cims(self):
        return self.actual_cims

    def get_cim(self, index):
        flat_index = self.indexes_converter(index)
        return self.actual_cims[flat_index]


"""sofc = SetOfCims('Z', [3, 3], 3)
sofc.build_actual_cims_structure()
print(sofc.actual_cims)
print(sofc.actual_cims[0,0])
print(sofc.actual_cims[1,2])
#print(sofc.indexes_converter([]))"""
