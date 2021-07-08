import json
import pandas as pd
import os
from abc import ABC, abstractmethod

class AbstractExporter(ABC):
    """Abstract class that exposes the methods to save in json format a network information
    along with one or more trajectories generated basing on it
    
    :param _variables: Dataframe containing the nodes labels and cardinalities
    :type _variables: pandas.DataFrame
    :param _dyn_str: Dataframe containing the structure of the network (edges)
    :type _dyn_str: pandas.DataFrame
    :param _dyn_cims: It contains, for every variable (label is the key), the SetOfCims object related to it
    :type _dyn_cims: dict
    :param _trajectories: List of trajectories, that can be added subsequently
    :type _trajectories: List
    """

    def __init__(self, variables: pd.DataFrame = None, dyn_str: pd.DataFrame = None, dyn_cims: dict = None):
        self._variables = variables
        self._dyn_str = dyn_str
        self._dyn_cims = dyn_cims
        self._trajectories = []

    def add_trajectory(self, trajectory: list):
        """Add a new trajectory to the current list

        :param trajectory: The trajectory to add
        :type trajectory: pandas.DataFrame
        """

        self._trajectories.append(trajectory)

    @abstractmethod
    def out_file(self, filename):
        """Create a file in current directory and write on it the previously added data 
            (variables, dyn_str, dyn_cims and trajectories)

        :param filename: Name of the output file (it must include json extension)
        :type filename: string
        """