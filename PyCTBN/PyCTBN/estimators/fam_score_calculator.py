
# License: MIT License


import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from math import log

from scipy.special import loggamma
from random import choice

from ..structure_graph.set_of_cims import SetOfCims
from ..structure_graph.network_graph import NetworkGraph
from ..structure_graph.conditional_intensity_matrix import ConditionalIntensityMatrix


'''

'''


class FamScoreCalculator:
    """
    Has the task of calculating the FamScore of a node by using a Bayesian score function
    """

    def __init__(self):
        #np.seterr('raise')
        pass

    # region theta

    def marginal_likelihood_theta(self,
                        cims: ConditionalIntensityMatrix,
                        alpha_xu: float,
                        alpha_xxu: float):
        """
        Calculate the FamScore value of the node identified by the label node_id

        :param cims: np.array with all the node's cims
        :type cims: np.array
        :param alpha_xu: hyperparameter over the CTBN’s q parameters, default to 0.1
        :type alpha_xu: float
        :param alpha_xxu: distribuited hyperparameter over the CTBN’s theta parameters
        :type alpha_xxu: float

        :return: the value of the marginal likelihood over theta
        :rtype: float
        """
        return np.sum(
                        [self.variable_cim_xu_marginal_likelihood_theta(cim,
                                                                    alpha_xu,
                                                                    alpha_xxu)
                        for cim in cims])

    def variable_cim_xu_marginal_likelihood_theta(self,
                        cim: ConditionalIntensityMatrix,
                        alpha_xu: float,
                        alpha_xxu: float):
        """
        Calculate the value of the marginal likelihood over theta given a cim

        :param cim: A conditional_intensity_matrix object with the sufficient statistics
        :type cim: class:'ConditionalIntensityMatrix'
        :param alpha_xu: hyperparameter over the CTBN’s q parameters, default to 0.1
        :type alpha_xu: float
        :param alpha_xxu: distribuited hyperparameter over the CTBN’s theta parameters
        :type alpha_xxu: float

        :return: the value of the marginal likelihood over theta
        :rtype: float
        """

        'get cim length'
        values = len(cim._state_residence_times)

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
                    cim: ConditionalIntensityMatrix,
                    alpha_xu: float,
                    alpha_xxu: float):
        """
        Calculate the marginal likelihood on q of the node when assumes a specif value
        and a specif parents's assignment

        :param cim: A conditional_intensity_matrix object with the sufficient statistics
        :type cim: class:'ConditionalIntensityMatrix'
        :param alpha_xu: hyperparameter over the CTBN’s q parameters
        :type alpha_xu: float
        :param alpha_xxu: distribuited hyperparameter over the CTBN’s theta parameters
        :type alpha_xxu: float

        :return: the value of the marginal likelihood over theta when the node assumes a specif value
        :rtype: float
        """

        values = list(range(len(cim._state_residence_times)))

        'remove the index because of the x != x^ condition in the summation '
        values.remove(index)

        'uncomment for alpha xx not uniform'
        #alpha_xxu = alpha_xu * cim.state_transition_matrix[index,index_x_first] / cim.state_transition_matrix[index, index])

        return (loggamma(alpha_xu) - loggamma(alpha_xu + cim.state_transition_matrix[index, index])) \
                + \
                np.sum([self.single_internal_cim_xxu_marginal_likelihood_theta(
                                                                        cim.state_transition_matrix[index,index_x_first],
                                                                        alpha_xxu)
                for index_x_first in values])


    def single_internal_cim_xxu_marginal_likelihood_theta(self,
                                                M_xxu_suff_stats: float,
                                                alpha_xxu: float=1):
        """Calculate the second part of the marginal likelihood over theta formula
        
        :param M_xxu_suff_stats: value of the suffucient statistic M[xx'|u]
        :type M_xxu_suff_stats: float
        :param alpha_xxu: distribuited hyperparameter over the CTBN’s theta parameters
        :type alpha_xxu: float

        :return: the value of the marginal likelihood over theta when the node assumes a specif value
        :rtype: float
        """
        return loggamma(alpha_xxu+M_xxu_suff_stats) - loggamma(alpha_xxu)

    # endregion

    # region q

    def marginal_likelihood_q(self,
                        cims: np.array,
                        tau_xu: float=0.1,
                        alpha_xu: float=1):
        """
        Calculate the value of the marginal likelihood over q of the node identified by the label node_id
        
        :param cims: np.array with all the node's cims
        :type cims: np.array
        :param tau_xu: hyperparameter over the CTBN’s q parameters
        :type tau_xu: float
        :param alpha_xu: hyperparameter over the CTBN’s q parameters
        :type alpha_xu: float


        :return: the value of the marginal likelihood over q
        :rtype: float
        """

        return np.sum([self.variable_cim_xu_marginal_likelihood_q(cim, tau_xu, alpha_xu) for cim in cims])

    def variable_cim_xu_marginal_likelihood_q(self,
                        cim: ConditionalIntensityMatrix,
                        tau_xu: float=0.1,
                        alpha_xu: float=1):
        """
        Calculate the value of the marginal likelihood over q given a cim
        
        :param cim: A conditional_intensity_matrix object with the sufficient statistics
        :type cim: class:'ConditionalIntensityMatrix'
        :param tau_xu: hyperparameter over the CTBN’s q parameters
        :type tau_xu: float
        :param alpha_xu: hyperparameter over the CTBN’s q parameters
        :type alpha_xu: float


        :return: the value of the marginal likelihood over q
        :rtype: float
        """

        'get cim length'
        values=len(cim._state_residence_times)

        'compute the marginal likelihood for the current cim'
        return np.sum([
                    self.single_cim_xu_marginal_likelihood_q(
                                                cim.state_transition_matrix[index, index],
                                                cim._state_residence_times[index],
                                                tau_xu,
                                                alpha_xu)
                    for index in range(values)])


    def single_cim_xu_marginal_likelihood_q(self,
                        M_xu_suff_stats: float,
                        T_xu_suff_stats: float,
                        tau_xu: float=0.1,
                        alpha_xu: float=1):
        """
        Calculate the marginal likelihood on q of the node when assumes a specif value
        and a specif parents's assignment
        
        :param M_xu_suff_stats: value of the suffucient statistic M[x|u]
        :type M_xxu_suff_stats: float
        :param T_xu_suff_stats: value of the suffucient statistic T[x|u]
        :type T_xu_suff_stats: float
        :param cim: A conditional_intensity_matrix object with the sufficient statistics
        :type cim: class:'ConditionalIntensityMatrix'
        :param tau_xu: hyperparameter over the CTBN’s q parameters
        :type tau_xu: float
        :param alpha_xu: hyperparameter over the CTBN’s q parameters
        :type alpha_xu: float


        :return: the value of the marginal likelihood of the node when assumes a specif value
        :rtype: float
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
                tau_xu: float=0.1,
                alpha_xu: float=1):
        """
        Calculate the FamScore value of the node
        
        
        :param cims: np.array with all the node's cims
        :type cims: np.array
        :param tau_xu: hyperparameter over the CTBN’s q parameters, default to 0.1
        :type tau_xu: float, optional
        :param alpha_xu: hyperparameter over the CTBN’s q parameters, default to 1
        :type alpha_xu: float, optional


        :return: the FamScore value of the node
        :rtype: float
        """

        'calculate alpha_xxu as a uniform distribution'                                
        alpha_xxu = alpha_xu /(len(cims[0]._state_residence_times) - 1)

        return self.marginal_likelihood_q(cims,
                                    tau_xu,
                                    alpha_xu) \
               + \
               self.marginal_likelihood_theta(cims, 
                                        alpha_xu,
                                        alpha_xxu)
