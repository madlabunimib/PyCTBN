
import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from math import log

from scipy.special import gamma
from random import choice

import set_of_cims as soCims 
import network_graph as net_graph
import conditional_intensity_matrix as cim_class


'''
TODO: Parlare dell'idea di ciclare sulle cim senza filtrare
TODO: Parlare di gamma in scipy e math(overflow) 
TODO: Problema warning overflow
'''

class FamScoreCalculator:
    """
    Has the task of calculate the FamScore of a node
    """

    def __init__(self):
        pass

    def marginal_likelihood_theta(self, 
                            node_id: str, 
                            set_of_cims: soCims.SetOfCims,
                            graph:net_graph.NetworkGraph):
        """
       calculate the value of the marginal likelihood over theta of the node identified by the label node_id
        Parameters:
            node_id: the label of the node
        Returns:
            the value of the marginal likelihood over theta
        """
        return 2

    def marginal_likelihood_q(self,
                        cims: np.array,
                        tau_xu:float = 1,
                        alpha_xu:float = 1):
        """
        calculate the value of the marginal likelihood over q of the node identified by the label node_id
        Parameters:
            cims: np.array with all the node's cims,
            tau_xu: hyperparameter over the CTBN’s q parameters
            alpha_xu: hyperparameter over the CTBN’s q parameters
        Returns:
            the value of the marginal likelihood over q
        """
        return np.prod([self.variable_cim_xu_marginal_likelihood_q(cim,tau_xu,alpha_xu) for cim in cims])
    
    def variable_cim_xu_marginal_likelihood_q(self,
                        cim:cim_class.ConditionalIntensityMatrix,
                        tau_xu:float = 1,
                        alpha_xu:float = 1):
        'get cim length'
        values=len(cim.state_residence_times) 

        'compute the marginal likelihood for the current cim'
        return np.prod([
                    self.single_cim_xu_marginal_likelihood_q(
                                                cim.state_transition_matrix[index,index],
                                                cim.state_residence_times[index],
                                                tau_xu,
                                                alpha_xu)
                    for index in range(values)])    


    def single_cim_xu_marginal_likelihood_q(self,
                        M_suff_stats:float,
                        T_suff_stats:float,
                        tau_xu:float = 1,
                        alpha_xu:float = 1):
        """
        calculate the marginal likelihood of the node when assumes a specif value
        and a specif parents's assignment 
        Parameters:
            cims: np.array with all the node's cims,
            tau_xu: hyperparameter over the CTBN’s q parameters
            alpha_xu: hyperparameter over the CTBN’s q parameters
        Returns:
            the marginal likelihood of the node when assumes a specif value
        """
        print(M_suff_stats)
        return  (gamma(alpha_xu + M_suff_stats + 1)* (tau_xu**(alpha_xu+1))) \
                / \
                (gamma(alpha_xu + 1)*((tau_xu + T_suff_stats)**(alpha_xu + M_suff_stats + 1)))


    def get_fam_score(self,
                cims: np.array,
                tau_xu:float = 1,
                alpha_xu:float = 1,
                alpha_xxu:float = 1):
        """
        calculate the FamScore value of the node identified by the label node_id
        Parameters:
            cims: np.array with all the node's cims,
            tau_xu: hyperparameter over the CTBN’s q parameters
            alpha_xu: hyperparameter over the CTBN’s q parameters
            alpha_xxu: hyperparameter over the CTBN’s theta parameters
        Returns:
            the FamScore value of the node
        """
        return log(
                    self.marginal_likelihood_q(cims,tau_xu,alpha_xu)
                ) \
                + \
                log(
                    self.marginal_likelihood_theta(cims,tau_xu,alpha_xu,alpha_xxu)
                    )
