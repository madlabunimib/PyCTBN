import sys
sys.path.append('../')
import itertools
import json
import typing

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

import abc 

from estimators import structure_estimator as se



class Optimizer(abc.ABC):
    """
    Interface class for all the optimizer's child classes

    :param node_id: the node label
    :type node_id: string
    :param structure_estimator: A structureEstimator Object to predict the structure
    :type structure_estimator: class:'StructureEstimator'
    
    """

    def __init__(self, node_id:str, structure_estimator: se.StructureEstimator):
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
