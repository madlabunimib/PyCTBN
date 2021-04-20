import json
import pandas as pd
import os

class JsonExporter(object):
    def __init__(self, variables, dyn_str, dyn_cims):
        self._variables = variables
        self._dyn_str = dyn_str
        self._dyn_cims = dyn_cims
        self._trajectories = []

    def add_trajectory(self, trajectory: list):
        self._trajectories.append(trajectory)

    def out_json(self, filename):
        data = [{
            "dyn.str": self._dyn_str,
            "variables": self._variables,
            "dyn.cims": self._dyn_cims,
            "samples": self._trajectories
        }]

        path = os.getcwd()
        with open(path + "/" + filename, "w") as json_file:
            json.dump(data, json_file)