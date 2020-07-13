import set_of_cims as socim
import numpy as np


class AmalgamatedCims:
    """
    Aggrega un insieme di oggetti SetOfCims indicizzandoli a partire dal node_id della variabile:
    {X:SetofCimsX, Y:SetOfCimsY.......}
    """
    # list_of_vars_orders contiene tutte le liste con i parent ordinati secondo il valore indx
    def __init__(self, states_number_per_node, list_of_keys, list_of_parents_states_number):
        self.sets_of_cims = []
        self.init_cims_structure(list_of_keys, states_number_per_node, list_of_parents_states_number)
        #self.states_per_variable = states_number

    def init_cims_structure(self, keys, states_number_per_node, list_of_parents_states_number):
        #print(keys)
        #print(list_of_parents_states_number)
        for indx, key in enumerate(keys):
            self.sets_of_cims.append(
                socim.SetOfCims(key, list_of_parents_states_number[indx], states_number_per_node[indx]))


    def get_set_of_cims(self, node_indx):
        return self.sets_of_cims[node_indx]

    def get_cims_of_node(self, node_indx, cim_indx):
        return self.sets_of_cims[node_indx].get_cim(cim_indx)

    def get_vars_order(self, node):
        return self.actual_cims[node][1]

    def update_state_transition_for_matrix(self, node, which_matrix, element_indx):
        self.sets_of_cims[node].update_state_transition(which_matrix, element_indx)

    def update_state_residence_time_for_matrix(self, which_node, which_matrix, which_element, time):
        self.sets_of_cims[which_node].update_state_residence_time(which_matrix, which_element, time)
