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

    def __init__(self, node_id, parents_states_number, node_states_number, p_combs):
        self.node_id = node_id
        self.parents_states_number = parents_states_number
        self.node_states_number = node_states_number
        self.actual_cims = []
        self.state_residence_times = None
        self.transition_matrices = None
        self.p_combs = p_combs
        self.build_actual_cims_structure()

    def build_actual_cims_structure(self):
        if not self.parents_states_number:
            self.state_residence_times = np.zeros((1, self.node_states_number), dtype=np.float)
            self.transition_matrices = np.zeros((1,self.node_states_number, self.node_states_number), dtype=np.int)
        else:
            self.state_residence_times = \
                np.zeros((np.prod(self.parents_states_number), self.node_states_number), dtype=np.float)
            self.transition_matrices = np.zeros([np.prod(self.parents_states_number), self.node_states_number,
                                                 self.node_states_number], dtype=np.int)


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
            cim_to_add = cim.ConditionalIntensityMatrix(state_res_time_vector, transition_matrix)
            cim_to_add.compute_cim_coefficients()
            #print(cim_to_add)
            self.actual_cims.append(cim_to_add)
        self.actual_cims = np.array(self.actual_cims)
        self.transition_matrices = None
        self.state_residence_times = None

    def get_cims(self):
        return self.actual_cims

    def get_cim(self, index):
        flat_index = self.indexes_converter(index)
        return self.actual_cims[flat_index]

    def filter_cims_with_mask(self, mask_arr, comb):
        if mask_arr.size <= 1:
            return self.actual_cims
        else:
            tmp_parents_comb_from_ids = np.argwhere(np.all(self.p_combs[:, mask_arr] == comb, axis=1)).ravel()
            #print("CIMS INDEXES TO USE!",tmp_parents_comb_from_ids)
            return self.actual_cims[tmp_parents_comb_from_ids]

"""sofc = SetOfCims('Z', [3, 3], 3)
sofc.build_actual_cims_structure()
print(sofc.actual_cims)
print(sofc.actual_cims[0,0])
print(sofc.actual_cims[1,2])
#print(sofc.indexes_converter([]))"""
