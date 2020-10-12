import sys
sys.path.append('../')


import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from math import log

from scipy.special import loggamma
from random import choice

import structure_graph.set_of_cims as soCims
import structure_graph.network_graph as net_graph
import structure_graph.conditional_intensity_matrix as cim_class


'''
TODO: Parlare dell'idea di ciclare sulle cim senza filtrare
TODO: Parlare del problema con gamma in scipy e math(overflow)
TODO: Problema warning overflow durante l'esecuzione
'''


class FamScoreCalculator:
    """
    Has the task of calculating the FamScore of a node
    """

    def __init__(self):
        np.seterr('raise')
        pass

    # region theta

    def marginal_likelihood_theta(self,
                        cims: cim_class.ConditionalIntensityMatrix,
                        alpha_xu: float = 1,
                        alpha_xxu: float = 1):
        """
        calculate the FamScore value of the node identified by the label node_id
        Parameters:
            cims: np.array with all the node's cims,
            alpha_xu: hyperparameter over the CTBN’s q parameters
            alpha_xxu: hyperparameter over the CTBN’s theta parameters
        Returns:
            the value of the marginal likelihood over theta
        """
        return np.sum(
                        [self.variable_cim_xu_marginal_likelihood_theta(cim,
                                                                    alpha_xu,
                                                                    alpha_xxu)
                        for cim in cims])

    def variable_cim_xu_marginal_likelihood_theta(self,
                        cim: cim_class.ConditionalIntensityMatrix,
                        alpha_xu: float = 1,
                        alpha_xxu: float = 1):
        """
        calculate the value of the marginal likelihood over theta given a cim
        Parameters:
            cim: A conditional_intensity_matrix object with the sufficient statistics,
            alpha_xu: hyperparameter over the CTBN’s q parameters
            alpha_xxu: hyperparameter over the CTBN’s theta parameters
        Returns:
            the value of the marginal likelihood over theta
        """

        'get cim length'
        values = len(cim.state_residence_times)

        'compute the marginal likelihood for the current cim'
        return np.sum([
                    self.single_cim_xu_marginal_likelihood_theta(
                                                index,
                                                cim,
                                                alpha_xu,
                                                alpha_xxu)
                    for index in range(values)])

    def single_cim_xu_marginal_likelihood_theta(self,
                    index: int,
                    cim: cim_class.ConditionalIntensityMatrix,
                    alpha_xu: float = 1,
                    alpha_xxu: float = 1):
        """
        calculate the marginal likelihood on q of the node when assumes a specif value
        and a specif parents's assignment
        Parameters:
            index: current x instance's index
            cim: A conditional_intensity_matrix object with the sufficient statistics,
            alpha_xu: hyperparameter over the CTBN’s q parameters
            alpha_xxu: hyperparameter over the CTBN’s theta parameters
        Returns:
            the marginal likelihood of the node when assumes a specif value
        """

        values = list(range(len(cim.state_residence_times)))

        'remove the index because of the x != x^ condition in the summation '
        values.remove(index)

        return (loggamma(alpha_xu) - loggamma(alpha_xu + cim.state_transition_matrix[index, index])) \
                + \
                np.sum([self.single_internal_cim_xxu_marginal_likelihood_theta(
                                                                        cim.state_transition_matrix[index,index_x_first],
                                                                        alpha_xxu)
                for index_x_first in values])


    def single_internal_cim_xxu_marginal_likelihood_theta(self,
                                                M_xxu_suff_stats: float,
                                                alpha_xxu: float=1):
        """
        calculate the second part of the marginal likelihood over theta formula
        Parameters:
            M_xxu_suff_stats: value of the suffucient statistic M[xx'|u]
            alpha_xxu: hyperparameter over the CTBN’s theta parameters
        Returns:
            the marginal likelihood of the node when assumes a specif value
        """
        return loggamma(alpha_xxu+M_xxu_suff_stats) - loggamma(alpha_xxu)

    # endregion

    # region q

    def marginal_likelihood_q(self,
                        cims: np.array,
                        tau_xu: float=1,
                        alpha_xu: float=1):
        """
        calculate the value of the marginal likelihood over q of the node identified by the label node_id
        Parameters:
            cims: np.array with all the node's cims,
            tau_xu: hyperparameter over the CTBN’s q parameters
            alpha_xu: hyperparameter over the CTBN’s q parameters
        Returns:
            the value of the marginal likelihood over q
        """
        return np.sum([self.variable_cim_xu_marginal_likelihood_q(cim, tau_xu, alpha_xu) for cim in cims])

    def variable_cim_xu_marginal_likelihood_q(self,
                        cim: cim_class.ConditionalIntensityMatrix,
                        tau_xu: float=1,
                        alpha_xu: float=1):
        """
        calculate the value of the marginal likelihood over q given a cim
        Parameters:
            cim: A conditional_intensity_matrix object with the sufficient statistics,
            tau_xu: hyperparameter over the CTBN’s q parameters
            alpha_xu: hyperparameter over the CTBN’s q parameters
        Returns:
            the value of the marginal likelihood over q
        """

        'get cim length'
        values=len(cim.state_residence_times)

        'compute the marginal likelihood for the current cim'
        return np.sum([
                    self.single_cim_xu_marginal_likelihood_q(
                                                cim.state_transition_matrix[index, index],
                                                cim.state_residence_times[index],
                                                tau_xu,
                                                alpha_xu)
                    for index in range(values)])


    def single_cim_xu_marginal_likelihood_q(self,
                        M_xu_suff_stats: float,
                        T_xu_suff_stats: float,
                        tau_xu: float=1,
                        alpha_xu: float=1):
        """
        calculate the marginal likelihood on q of the node when assumes a specif value
        and a specif parents's assignment
        Parameters:
            M_xu_suff_stats: value of the suffucient statistic M[x|u]
            T_xu_suff_stats: value of the suffucient statistic T[x|u]
            tau_xu: hyperparameter over the CTBN’s q parameters
            alpha_xu: hyperparameter over the CTBN’s q parameters
        Returns:
            the marginal likelihood of the node when assumes a specif value
        """
        return (
                loggamma(alpha_xu + M_xu_suff_stats + 1) + 
                                                        (log(tau_xu)
                                                        *
                                                        (alpha_xu+1))
                ) \
                - \
                (loggamma(alpha_xu + 1)+(
                                    log(tau_xu + T_xu_suff_stats) 
                                    *
                                    (alpha_xu + M_xu_suff_stats + 1))
                )

    # end region

    def get_fam_score(self,
                cims: np.array,
                tau_xu: float=1,
                alpha_xu: float=1,
                alpha_xxu: float=1):
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
        return self.marginal_likelihood_q(cims,
                                    tau_xu,
                                    alpha_xu) \
               + \
               self.marginal_likelihood_theta(cims, 
                                        alpha_xu,
                                        alpha_xxu)
