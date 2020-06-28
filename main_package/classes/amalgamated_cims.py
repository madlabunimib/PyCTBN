import set_of_cims as socim
import numpy as np


class AmalgamatedCims:

    def __init__(self, states_number,list_of_keys, nodes_value, list_of_vars_order):
        self.actual_cims = {}
        self.init_cims_structure(list_of_keys, nodes_value, list_of_vars_order)
        self.states_per_variable = states_number

    def init_cims_structure(self, keys, nodes_val, list_of_vars_order):
        for key, vars_order in (keys, list_of_vars_order):
            self.actual_cims[key] = socim.SetOfCims(key, vars_order, nodes_val)


    def compute_matrix_indx(self, row, col):
        return self.state_per_variable * row + col

    def get_vars_order(self, node):
        return self.actual_cims[node][1]

    def update_state_transition_for_matrix(self, node, dict_of_indxs, element_indx):
        self.actual_cims[node]
