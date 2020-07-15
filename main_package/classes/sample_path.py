import pandas as pd
import numpy as np
import os
import json_importer as imp
import trajectory as tr
import structure as st


class SamplePath:
    """
    Contiene l'aggregazione di una o più traiettorie e la struttura della rete.
    Ha il compito dato di costruire tutte gli oggetti Trajectory e l'oggetto Structure
    a partire dai dataframe contenuti in self.importer


    :importer: l'oggetto Importer che ha il compito di caricare i dataset
    :trajectories: lista di oggetti Trajectories
    :structure: oggetto Structure
    """

    def __init__(self, files_path):
        self.importer = imp.JsonImporter(files_path)
        self._trajectories = None
        self._structure = None

    def build_trajectories(self):
        self.importer.import_data()
        self._trajectories = \
            tr.Trajectory(self.importer.build_list_of_samples_array(self.importer.concatenated_samples),
                          len(self.importer.sorter) + 1)
        #self.trajectories.append(trajectory)
        self.importer.clear_data_frames()

    def build_structure(self):
        self._structure = st.Structure(self.importer.structure, self.importer.variables)

    @property
    def trajectories(self):
        return self._trajectories

    @property
    def structure(self):
        return self._structure

"""os.getcwd()
os.chdir('..')
path = os.getcwd() + '/data'


os.getcwd()
os.chdir('..')
path = os.getcwd() + '/data'

s1 = SamplePath(path)
s1.build_trajectories()
s1.build_structure()
print(s1.trajectories[0].get_complete_trajectory())"""

