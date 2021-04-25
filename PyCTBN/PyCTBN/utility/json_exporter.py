import json
import pandas as pd
import os

class JsonExporter(object):
    """Provides the methods to save in json format a network information
    along with one or more trajectories generated basing on it
    
    :param _variables: List of dictionaries, representing the variables in the network and their cardinality
    :type _variables: List
    :param _dyn_str: List of dictionaries, each of which represents an edge ({"From": "", "To": ""})
    :type _dyn_str: List
    :param _dyn_cims: It contains, for every variable (label is the key), the CIM values related to it
    :type _dyn_cims: Dict
    :param _trajectories: List of trajectories, that can be added subsequently
    :type _trajectories: List
    """

    def __init__(self, variables, dyn_str, dyn_cims):
        self._variables = variables
        self._dyn_str = dyn_str
        self._dyn_cims = dyn_cims
        self._trajectories = []

    def add_trajectory(self, trajectory: list):
        """Add a new trajectory to the current list

        :param trajectory: The trajectory to add. It must already be in json form, and not as pandas.DataFrame
        :type trajectory: List
        """

        self._trajectories.append(trajectory)

    def out_json(self, filename):
        """Create a file in current directory and write on it the previously added data 
            (variables, dyn_str, dyn_cims and trajectories)

        :param filename: Name of the output file (it must include json extension)
        :type filename: string
        """

        data = [{
            "dyn.str": self._dyn_str,
            "variables": self._variables,
            "dyn.cims": self._dyn_cims,
            "samples": self._trajectories
        }]

        path = os.getcwd()
        with open(path + "/" + filename, "w") as json_file:
            json.dump(data, json_file)