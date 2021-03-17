from ..utility.abstract_importer import AbstractImporter
from .conditional_intensity_matrix import ConditionalIntensityMatrix
from .set_of_cims import SetOfCims
from .trajectory import Trajectory
import numpy as np
import pandas as pd
import re
from numpy import random

class TrajectoryGenerator(object):
    def __init__(self, importer: AbstractImporter):
        self._importer = importer
        self._importer.import_data(0)

        self._vnames = self._importer._df_variables.iloc[:, 0].to_list()
        
        self._parents = {}
        for v in self._vnames:
            self._parents[v] = self._importer._df_structure.where(self._importer._df_structure["To"] == v).dropna()["From"].tolist()

        self._cims = {}
        sampled_cims = self._importer._raw_data[0]["dyn.cims"]
        for v in sampled_cims.keys():
            p_combs = []
            v_cims = []
            for comb in sampled_cims[v].keys():
                p_combs.append(np.array(re.findall(r"=(\d)", comb)).astype("int"))
                cim = pd.DataFrame(sampled_cims[v][comb]).to_numpy()      
                v_cims.append(ConditionalIntensityMatrix(cim = cim))
            
            sof = SetOfCims(node_id = v, parents_states_number = [self._importer._df_variables.where(self._importer._df_variables["Name"] == p)["Value"] for p in self._parents[v]], 
                node_states_number = self._importer._df_variables.where(self._importer._df_variables["Name"] == v)["Value"], p_combs = p_combs, cims = v_cims)
            self._cims[v] = sof

    def CTBN_Sample(self, t_end):
        #Replace v_values with sigma (pandas dataframe)
        t = 0
        sigma = pd.DataFrame(columns = (["Time"] + self._vnames))
        sigma.loc[len(sigma)] = [0] + [0 for v in self._vnames]
        time = np.full(len(self._vnames), np.NaN)

        while True:
            for i in range(0, time.size):
                if np.isnan(time[i]):
                    # Probability to transition from current state v_values[i] to (1 - v_values[i])
                    current_values = sigma.loc[len(sigma) - 1]
                    cim = self._cims[self._vnames[i]].filter_cims_with_mask(np.array([True for p in self._parents[self._vnames[i]]]), 
                        [current_values.at[p] for p in self._parents[self._vnames[i]]])[0].cim
                    param = cim[current_values.at[self._vnames[i]]][1 - current_values.at[self._vnames[i]]]

                    time[i] = t + random.exponential(scale = param)
        
            # next = index of the variable that will transition first
            next = time.argmin()
            t = time[next]

            if t >= t_end:
                return sigma
            else:
                new_row = pd.DataFrame(sigma[-1:].values, columns = sigma.columns)
                new_row.loc[0].at[self._vnames[next]] = 1 - new_row.loc[0].at[self._vnames[next]]
                new_row.loc[0].at["Time"] = round(t, 4)
                sigma = sigma.append(new_row, ignore_index = True)

                """ sup = sigma.loc[len(sigma) - 1]
                sup.at[self._vnames[next]] = 1 - sup.at[self._vnames[next]]
                sup.at["Time"] = round(t, 4)
                sigma.loc[len(sigma)] = sup """

                # undefine variable time
                time[next] = np.NaN