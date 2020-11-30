import set_of_cims as socim


class SetsOfCimsContainer:
    """
    Aggrega un insieme di oggetti SetOfCims
    """
    def __init__(self, list_of_keys, states_number_per_node, list_of_parents_states_number, p_combs_list):
        self.sets_of_cims = None
        self.init_cims_structure(list_of_keys, states_number_per_node, list_of_parents_states_number, p_combs_list)
        #self.states_per_variable = states_number

    def init_cims_structure(self, keys, states_number_per_node, list_of_parents_states_number, p_combs_list):
        """for indx, key in enumerate(keys):
            self.sets_of_cims.append(
                socim.SetOfCims(key, list_of_parents_states_number[indx], states_number_per_node[indx]))"""
        self.sets_of_cims = [socim.SetOfCims(pair[1], list_of_parents_states_number[pair[0]], states_number_per_node[pair[0]], p_combs_list[pair[0]])
                             for pair in enumerate(keys)]

    def get_set_of_cims(self, node_indx):
        return self.sets_of_cims[node_indx]

    def get_cims_of_node(self, node_indx, cim_indx):
        return self.sets_of_cims[node_indx].get_cim(cim_indx)

