import json
import pandas as pd
import numpy as np
import os

from .abstract_exporter import AbstractExporter

class JsonExporter(AbstractExporter):
    """Provides the methods to save in json format a network information
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

    def out_file(self, filename):
        """Create a file in current directory and write on it the previously added data 
            (variables, dyn_str, dyn_cims and trajectories)

        :param filename: Name of the output file (it must include json extension)
        :type filename: string
        """

        data = [{
            "dyn.str": json.loads(self._dyn_str.to_json(orient="records")),
            "variables": json.loads(self._variables.to_json(orient="records")),
            "dyn.cims": self.cims_to_json(),
            "samples": [json.loads(trajectory.to_json(orient="records")) for trajectory in self._trajectories]
        }]

        path = os.getcwd()
        with open(path + "/" + filename, "w") as json_file:
            json.dump(data, json_file)

    """Restructure the CIMs object in order to fit the standard JSON file structure 
    """
    def cims_to_json(self) -> dict:
        json_cims = {}

        for i, l in enumerate(self._variables.iloc[:, 0].to_list()):
            json_cims[l] = {}
            parents = self._dyn_str.where(self._dyn_str["To"] == l).dropna()["From"].tolist()
            for j, comb in enumerate(self._dyn_cims[l].p_combs):
                comb_key = ""
                if len(parents) != 0:
                    for k, val in enumerate(comb):
                        comb_key += parents[k] + "=" + str(val)
                        if k < len(comb) - 1:
                            comb_key += ","
                else:
                    comb_key = l

                cim = self._dyn_cims[l].filter_cims_with_mask(np.array([True for p in parents]), comb)
                if len(parents) == 1:
                    cim = cim[comb[0]].cim
                elif len(parents) == 0:
                    cim = cim[0].cim
                else:
                    cim = cim[0].cim
                
                json_cims[l][comb_key] = [dict([(str(i), val) for i, val in enumerate(row)]) for row in cim]

        return json_cims