import set_of_cims as socim
import numpy as np


class AmalgamatedCims:

    def __init__(self, states_number,list_of_keys, list_of_vars_order):
        self.sets_of_cims = {}
        self.init_cims_structure(list_of_keys, states_number, list_of_vars_order)
        self.states_per_variable = states_number

    def init_cims_structure(self, keys, nodes_val, list_of_vars_order):
        print(keys)
        print(list_of_vars_order)
        for indx, key in enumerate(keys):
            self.sets_of_cims[key] = socim.SetOfCims(key, list_of_vars_order[indx], nodes_val)



    def get_set_of_cims(self, node_id):
        return self.sets_of_cims[node_id]

    def get_vars_order(self, node):
        return self.actual_cims[node][1]

    def update_state_transition_for_matrix(self, node, dict_of_indxs, element_indx):
        self.actual_cims[node]
