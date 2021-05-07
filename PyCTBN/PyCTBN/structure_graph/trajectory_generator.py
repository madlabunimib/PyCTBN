from ..utility.abstract_importer import AbstractImporter
from .conditional_intensity_matrix import ConditionalIntensityMatrix
from .set_of_cims import SetOfCims
from .trajectory import Trajectory
import numpy as np
import pandas as pd
import re
import json
from numpy import random
from multiprocessing import Process, Manager

class TrajectoryGenerator(object):
    """Provides the methods to generate a trajectory basing on the network defined
    in the importer.
    
    :param _importer: the Importer object which contains the imported and processed data
    :type _importer: AbstractImporter
    :param _vnames: List of the variables labels that belong to the network
    :type _vnames: List
    :param _parents: It contains, for each variable label (the key), the list of related parents labels
    :type _parents: Dict
    :param _cims: It contains, for each variable label (the key), the SetOfCims object related to it
    :type _cims: Dict
    :param _generated_trajectory: Result of the execution of CTBN_Sample, contains the output trajectory
    :type _generated_trajectory: pandas.DataFrame
    """

    def __init__(self, importer: AbstractImporter = None, variables: list = None, dyn_str: list = None, dyn_cims: dict = None):
        """Constructor Method
            It parses and elaborates the data fetched from importer (if defined, otherwise variables, dyn_str and dyn_cims are used) 
            in order to make the objects structure more suitable for the forthcoming trajectory generation
        """
        
        self._importer = importer

        self._vnames = self._importer._df_variables.iloc[:, 0].to_list() if importer is not None else [v["Name"] for v in variables]
        
        self._parents = {}
        for v in self._vnames:
            if importer is not None:
                self._parents[v] = self._importer._df_structure.where(self._importer._df_structure["To"] == v).dropna()["From"].tolist()
            else:
                self._parents[v] = [edge["From"] for edge in dyn_str if edge["To"] == v]

        self._cims = {}
        sampled_cims = self._importer._raw_data[0]["dyn.cims"] if importer is not None else dyn_cims
        for v in sampled_cims.keys():
            p_combs = []
            v_cims = []
            for comb in sampled_cims[v].keys():
                p_combs.append(np.array(re.findall(r"=(\d)", comb)).astype("int"))
                cim = pd.DataFrame(sampled_cims[v][comb]).to_numpy()    
                v_cims.append(ConditionalIntensityMatrix(cim = cim))

            if importer is not None:
                sof = SetOfCims(node_id = v, parents_states_number = [self._importer._df_variables.where(self._importer._df_variables["Name"] == p)["Value"] for p in self._parents[v]], 
                    node_states_number = self._importer._df_variables.where(self._importer._df_variables["Name"] == v)["Value"], p_combs = np.array(p_combs), cims = v_cims)
            else:
                sof = SetOfCims(node_id = v, parents_states_number = [[variable["Value"] for variable in variables if variable["Name"] == p][0] for p in self._parents[v]], 
                    node_states_number = [variable for variable in variables if variable["Name"] == v][0]["Value"], p_combs = np.array(p_combs), cims = v_cims)
            self._cims[v] = sof

    def CTBN_Sample(self, t_end = -1, max_tr = -1):
        """This method implements the generation of a trajectory, basing on the network structure and
            on the coefficients defined in the CIMs.
            The variables are initialized with value 0, and the method takes care of adding the
            conventional last row made up of -1.

        :param t_end: If defined, the sampling ends when end time is reached
        :type t_end: float
        :param max_tr: Parameter taken in consideration in case that t_end isn't defined. It specifies the number of transitions to execute
        :type max_tr: int
        """

        t = 0
        sigma = pd.DataFrame(columns = (["Time"] + self._vnames))
        sigma.loc[len(sigma)] = [0] + [0 for v in self._vnames]
        time = np.full(len(self._vnames), np.NaN)
        n_tr = 0
        self._generated_trajectory = None

        while True:
            current_values = sigma.loc[len(sigma) - 1]
            
            for i in range(0, time.size):
                if np.isnan(time[i]):
                    n_parents = len(self._parents[self._vnames[i]])
                    cim_obj = self._cims[self._vnames[i]].filter_cims_with_mask(np.array([True for p in self._parents[self._vnames[i]]]), 
                        [current_values.at[p] for p in self._parents[self._vnames[i]]])

                    if n_parents == 1:
                        cim = cim_obj[current_values.at[self._parents[self._vnames[i]][0]]].cim
                    else:
                        cim = cim_obj[0].cim

                    param = -1 * cim[current_values.at[self._vnames[i]]][current_values.at[self._vnames[i]]]

                    time[i] = t + random.exponential(scale = param)
        
            # next = index of the variable that will transition first
            next = time.argmin()
            t = time[next]

            if (max_tr != -1 and n_tr == max_tr) or (t_end != -1 and t >= t_end):
                sigma.loc[len(sigma) - 1, self._vnames] = -1
                self._generated_trajectory = sigma
                return sigma
            else:
                n_parents = len(self._parents[self._vnames[next]])
                cim_obj = self._cims[self._vnames[next]].filter_cims_with_mask(np.array([True for p in self._parents[self._vnames[next]]]), 
                    [current_values.at[p] for p in self._parents[self._vnames[next]]])

                if n_parents == 1:
                    cim = cim_obj[current_values.at[self._parents[self._vnames[next]][0]]].cim
                else:
                    cim = cim_obj[0].cim
                    
                cim_row = np.array(cim[current_values.at[self._vnames[next]]])
                cim_row[current_values.at[self._vnames[next]]] = 0
                cim_row /= sum(cim_row)
                rand_mult = np.random.multinomial(1, cim_row, size=1)

                new_row = pd.DataFrame(sigma[-1:].values, columns = sigma.columns)
                new_row.loc[0].at[self._vnames[next]] = np.where(rand_mult[0] == 1)[0][0]
                new_row.loc[0].at["Time"] = round(t, 4)
                sigma = sigma.append(new_row, ignore_index = True)

                n_tr += 1

                # undefine variable time
                time[next] = np.NaN
                for i, v in enumerate(self._parents):
                    if self._vnames[next] in self._parents[v]:
                        time[i] = np.NaN

    def worker(self, t_end, max_tr, trajectories):
        """Single process that will be executed in parallel in order to generate one trajectory. 

            :param t_end: If defined, the sampling ends when end time is reached
            :type t_end: float
            :param max_tr: Parameter taken in consideration in case that t_end isn't defined. It specifies the number of transitions to execute
            :type max_tr: int
            :param trajectories: Shared list that contains to which the generated trajectory is added
            :type trajectories: list
        """
        
        trajectory = self.CTBN_Sample(t_end = t_end, max_tr = max_tr)
        trajectories.append(trajectory)

    def multi_trajectory(self, t_ends: list = None, max_trs: list = None):
        """Generate n trajectories in parallel, where n is the number of items in 
            t_ends, if defined, or the number of items in max_trs otherwise

            :param t_ends: List of t_end values for the trajectories that will be generated
            :type t_ends: list
            :param max_trs: List of max_tr values for the trajectories that will be generated
            :type max_trs: list
        """

        if t_ends is None and max_trs is None:
            return

        trajectories = Manager().list()

        if t_ends is not None:
            processes = [Process(target = self.worker, args = (t, -1, trajectories)) for t in t_ends]
        else:   
            processes = [Process(target = self.worker, args = (-1, m, trajectories)) for m in max_trs]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return trajectories

    def to_json(self):
        """Convert the last generated trajectory from pandas.DataFrame object type to JSON format
            (suitable to do input/output of the trajectory with file)
        """

        return json.loads(self._generated_trajectory.to_json(orient="records"))