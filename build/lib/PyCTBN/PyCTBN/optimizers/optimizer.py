
import itertools
import json
import typing

import networkx as nx
import numpy as np

import abc 

from ..estimators.structure_estimator import StructureEstimator



class Optimizer(abc.ABC):
    """
    Interface class for all the optimizer's child PyCTBN

    :param node_id: the node label
    :type node_id: string
    :param structure_estimator: A structureEstimator Object to predict the structure
    :type structure_estimator: class:'StructureEstimator'
    
    """

    def __init__(self, node_id:str, structure_estimator: StructureEstimator):
        self.node_id = node_id
        self.structure_estimator = structure_estimator
        

    @abc.abstractmethod
    def optimize_structure(self) -> typing.List:
        """
        Compute Optimization process for a structure_estimator

        :return: the estimated structure for the node
        :rtype: List
        """
        pass
