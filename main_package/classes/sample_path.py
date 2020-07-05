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
        print()
        self.importer = imp.JsonImporter(files_path)
        self.trajectories = []
        self.structure = None

    def build_trajectories(self):
        self.importer.import_data()
        #for traj_data_frame in self.importer.df_samples_list:
        trajectory = tr.Trajectory(self.importer.build_list_of_samples_array(self.importer.concatenated_samples))
        self.trajectories.append(trajectory)
        self.importer.clear_data_frames()

    def build_structure(self):
        self.structure = st.Structure(self.importer.df_structure, self.importer.df_variables)

    def get_number_trajectories(self):
        return len(self.trajectories)



