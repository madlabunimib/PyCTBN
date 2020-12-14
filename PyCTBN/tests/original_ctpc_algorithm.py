
import json
from itertools import combinations
import typing

import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist
from scipy.stats import f as f_dist
from tqdm import tqdm

from PyCTBN.classes.abstract_importer import AbstractImporter


class OriginalCTPCAlgorithm(AbstractImporter):
    """
    Implements the abstracts methods of AbstractImporter and adds all the necessary methods to process and prepare the data in json ext.
    with the following structure:
    [0]
        |_ dyn.cims
        |_ dyn.str
        |_ samples
        |_ variabels
    :_file_path: the path of the file that contains tha data to be imported
    :_samples_label: the reference key for the samples in the trajectories
    :_structure_label: the reference key for the structure of the network data
    :_variables_label: the reference key for the cardinalites of the nodes data
    :_time_key: the key used to identify the timestamps in each trajectory
    :_variables_key: the key used to identify the names of the variables in the net
    :_df_samples_list: a Dataframe list in which every df contains a trajectory
    """

    def dataset_id(self) -> object:
        pass

    def __init__(self, file_path: str, samples_label: str, structure_label: str, variables_label: str, time_key: str,
                 variables_key: str, raw_data: typing.List):
        """
        Parameters:
            file_path: the path of the file that contains tha data to be imported
            :_samples_label: the reference key for the samples in the trajectories
            :_structure_label: the reference key for the structure of the network data
            :_variables_label: the reference key for the cardinalites of the nodes data
            :_time_key: the key used to identify the timestamps in each trajectory
            :_variables_key: the key used to identify the names of the variables in the net
        """
        self.samples_label = samples_label
        self.structure_label = structure_label
        self.variables_label = variables_label
        self.time_key = time_key
        self.variables_key = variables_key
        self.df_samples_list = None
        self.trajectories = None
        self._array_indx  = None
        self.matrix = None
        super(OriginalCTPCAlgorithm, self).__init__(file_path)
        self._raw_data = raw_data

    def import_data(self, indx):
        """
        Imports and prepares all data present needed for subsequent processing.
        Parameters:
            :void
        Returns:
            _void
        """
        self._array_indx = indx
        self.df_samples_list = self.import_trajectories(self._raw_data)
        self._sorter = self.build_sorter(self.df_samples_list[0])
        #self.compute_row_delta_in_all_samples_frames(self._df_samples_list)
        #self.clear_data_frame_list()
        self._df_structure = self.import_structure(self._raw_data)
        self._df_variables = self.import_variables(self._raw_data, self._sorter)

    def datasets_numb(self):
        return len(self._raw_data)

    def import_trajectories(self, raw_data: typing.List):
        """
        Imports the trajectories in the list of dicts raw_data.
        Parameters:
            :raw_data: List of Dicts
        Returns:
            :List of dataframes containing all the trajectories
        """
        return self.normalize_trajectories(raw_data, self._array_indx, self.samples_label)

    def import_structure(self, raw_data: typing.List) -> pd.DataFrame:
        """
        Imports in a dataframe the data in the list raw_data at the key _structure_label

        Parameters:
            :raw_data: the data
        Returns:
            :Daframe containg the starting node a ending node of every arc of the network
        """
        return self.one_level_normalizing(raw_data, self._array_indx, self.structure_label)

    def import_variables(self, raw_data: typing.List, sorter: typing.List) -> pd.DataFrame:
        """
        Imports the data in raw_data at the key _variables_label.
        Sorts the row of the dataframe df_variables using the list sorter.

        Parameters:
            :raw_data: the data
            :sorter: the header of the dataset containing only variables symbolic labels
        Returns:
            :Datframe containg the variables simbolic labels and their cardinalities
        """
        return self.one_level_normalizing(raw_data, self._array_indx, self.variables_label)


    def read_json_file(self) -> typing.List:
        """
        Reads the JSON file in the path self.filePath

        Parameters:
              :void
        Returns:
              :data: the contents of the json file

        """
        with open(self._file_path) as f:
            data = json.load(f)
            return data

    def one_level_normalizing(self, raw_data: typing.List, indx: int, key: str) -> pd.DataFrame:
        """
        Extracts the one-level nested data in the list raw_data at the index indx at the key key

        Parameters:
            :raw_data: List of Dicts
            :indx: The index of the array from which the data have to be extracted
            :key: the key for the Dicts from which exctract data
        Returns:
            :a normalized dataframe:

        """
        return pd.DataFrame(raw_data[indx][key])

    def normalize_trajectories(self, raw_data: typing.List, indx: int, trajectories_key: str):
        """
        Extracts the traj in raw_data at the index index at the key trajectories key.

        Parameters:
            :raw_data: the data
            :indx: the index of the array from which extract data
            :trajectories_key: the key of the trajectories objects
        Returns:
            :A list of daframes containg the trajectories
        """
        dataframe = pd.DataFrame
        smps = raw_data[indx][trajectories_key]
        df_samples_list = [dataframe(sample) for sample in smps]
        return df_samples_list

    def build_sorter(self, sample_frame: pd.DataFrame) -> typing.List:
        """
        Implements the abstract method build_sorter for this dataset
        """
        columns_header = list(sample_frame.columns.values)
        columns_header.remove(self.time_key)
        return columns_header

    def clear_data_frame_list(self):
        """
        Removes all values present in the dataframes in the list _df_samples_list
        Parameters:
            :void
        Returns:
            :void
        """
        for indx in range(len(self.df_samples_list)):
            self.df_samples_list[indx] = self.df_samples_list[indx].iloc[0:0]


    def prepare_trajectories(self, trajectories, variables):
        """
            Riformula le traiettorie per rendere più efficiente la fase di computazione delle cim

            Parameters
            -------------
            trajectories: [pandas.DataFrame]
                Un array di pandas dataframe contenente tutte le traiettorie. Ogni array avrà una
                colonna per il timestamp (sempre la prima) e n colonne una per ogni variabili
                presente nella rete.
            variables: pandas.DataFrame
                Pandas dataframe contenente due colonne: il nome della variabile e cardinalità
                della variabile
        """
        dimensions = np.array([x.shape[0] - 1 for x in trajectories], dtype=np.int)
        ret_array = np.zeros([dimensions.sum(), trajectories[0].shape[1] * 2])
        cum_dim = np.zeros(len(trajectories) + 1, dtype=np.int)
        cum_dim[1:] = dimensions.cumsum()
        dimensions.cumsum()
        for it in range(len(trajectories)):
            tmp = trajectories[it].to_numpy()
            dim = tmp.shape[1]
            ret_array[cum_dim[it]:cum_dim[it + 1], 0:dim] = tmp[:-1]
            ret_array[cum_dim[it]:cum_dim[it + 1], dim] = np.diff(tmp[:, 0])
            ret_array[cum_dim[it]:cum_dim[it + 1], dim + 1:] = np.roll(tmp[:, 1:], -1, axis=0)[:-1]
        self.trajectories = ret_array
        #self.variables = variables

    @staticmethod
    def _compute_cim(trajectories, child_id, parents_id, T_vector, M_vector, parents_comb, M, T):
        """Funzione interna per calcolare le CIM

        Parameters:
        -----------
        trajectories: np.array
            Array contenente le traiettorie. (self.trajectories)

        child_id: int
            Indice del nodo di cui si vogliono calcolare le cim

        parents:id: [int]
            Array degli indici dei genitori nel nodo child_id

        T_vector: np.array
            Array numpy per l'indicizzazione dell'array T

        M_vector: np.array
            Array numpy per l'indicizzazione dell'array M

        parents_comb: [(int)]
            Array di tuple contenenti tutte le possibili combinazioni dei genitori di child_id

        M: np.array
            Array numpy contenente  la statistica sufficiente M

        T: np.array
            Array numpy contenente la statistica sufficiente T

        Returns:
        ---------
        CIM: np.array
            Array numpy contenente le CIM
        """
        #print(T)
        diag_indices = np.array([x * M.shape[1] + x % M.shape[1] for x in range(M.shape[0] * M.shape[1])],
                                dtype=np.int64)
        #print(diag_indices)
        T_filter = np.array([child_id, *parents_id], dtype=np.int) + 1
        #print("TFilter",T_filter)
        #print("TVector", T_vector)
        #print("Trajectories", trajectories)
        #print("Actual TVect",T_vector / T_vector[0])
        #print("Masked COlumns", trajectories[:, T_filter])  # Colonne non shiftate dei values
        #print("Masked Multiplied COlumns",trajectories[:, T_filter] * (T_vector / T_vector[0]) )
        #print("Summing",np.sum(trajectories[:, T_filter] * (T_vector / T_vector[0]), axis=1))
        #print("Deltas",trajectories[:, int(trajectories.shape[1] / 2)]) # i delta times
        assert np.sum(trajectories[:, T_filter] * (T_vector / T_vector[0]), axis=1).size == \
               trajectories[:, int(trajectories.shape[1] / 2)].size
        #print(T_vector[-1])
        T[:] = np.bincount(np.sum(trajectories[:, T_filter] * T_vector / T_vector[0], axis=1).astype(np.int), \
                           trajectories[:, int(trajectories.shape[1] / 2)], minlength=T_vector[-1]).reshape(-1,
                                                                                                            T.shape[1])
        #print("Shape", T.shape[1])
        #print(np.bincount(np.sum(trajectories[:, T_filter] * T_vector / T_vector[0], axis=1).astype(np.int), \
                           #trajectories[:, int(trajectories.shape[1] / 2)], minlength=T_vector[-1]))
        ###### Transitions #######

        #print("Shifted Node column", trajectories[:, int(trajectories.shape[1] / 2) + 1 + child_id].astype(np.int))
        #print("Step 2", trajectories[:, int(trajectories.shape[1] / 2) + 1 + child_id].astype(np.int) >= 0)
        trj_tmp = trajectories[trajectories[:, int(trajectories.shape[1] / 2) + 1 + child_id].astype(np.int) >= 0]
        #print("Trj Temp", trj_tmp)


        M_filter = np.array([child_id, child_id, *parents_id], dtype=np.int) + 1
        #print("MFilter", M_filter)
        M_filter[0] += int(trj_tmp.shape[1] / 2)
        #print("MFilter", M_filter)
        #print("MVector", M_vector)
        #print("Division", M_vector / M_vector[0])
        #print("Masked Traj temp", (trj_tmp[:, M_filter]))
        #print("Masked Multiplied Traj temp", trj_tmp[:, M_filter] * M_vector / M_vector[0])
        #print("Summing", np.sum(trj_tmp[:, M_filter] * M_vector / M_vector[0], axis=1))
        #print(M.shape[2])

        M[:] = np.bincount(np.sum(trj_tmp[:, M_filter] * M_vector / M_vector[0], axis=1).astype(np.int), \
                           minlength=M_vector[-1]).reshape(-1, M.shape[1], M.shape[2])
        #print("M!!!!!!!", M)
        M_raveled = M.ravel()
        #print("Raveled", M_raveled)
        M_raveled[diag_indices] = 0
        M_raveled[diag_indices] = np.sum(M, axis=2).ravel()
        #print("Raveled", M_raveled)
        q = (M.ravel()[diag_indices].reshape(-1, M.shape[1]) + 1) / (T + 1)
        theta = (M + 1) / (M.ravel()[diag_indices].reshape(-1, M.shape[2], 1) + 1)
        negate_main_diag = np.ones((M.shape[1], M.shape[2]))
        np.fill_diagonal(negate_main_diag, -1)
        theta = np.multiply(theta, negate_main_diag)
        return theta * q.reshape(-1, M.shape[2], 1)

    def compute_cim(self, child_id, parents_id):
        """Metodo utilizzato per calcolare le CIM di un nodo dati i suoi genitori

        Parameters:
        -----------
        child_id: int
            Indice del nodo di cui si vogliono calcolare le cim

        parents:id: [int]
            Array degli indici dei genitori nel nodo child_id

        Return:
        ----------
        Restituisce una tupla contenente:
            parents_comb: [(int)]
                Array di tuple contenenti tutte le possibili combinazioni dei genitori di child_id

            M: np.array
                Array numpy contenente  la statistica sufficiente M

            T: np.array
                Array numpy contenente la statistica sufficiente T

            CIM: np.array
                Array numpy contenente le CIM

        """
        tmp = []
        child_id = int(child_id)
        parents_id = np.array(parents_id, dtype=np.int)
        parents_id.sort()
        #print("Parents id",parents_id)
        #breakpoint()
        for idx in parents_id:
            tmp.append([x for x in range(self.variables.loc[idx, "Value"])])
        #print("TIMP", tmp)
        if len(parents_id) > 0:
            parents_comb = np.array(np.meshgrid(*tmp)).T.reshape(-1, len(parents_id))
            #print(np.argsort(parents_comb))
            #print("PArents COmb", parents_comb)
            if len(parents_id) > 1:
                tmp_comb = parents_comb[:, 1].copy()
                #print(tmp_comb)
                parents_comb[:, 1] = parents_comb[:, 0].copy()
                parents_comb[:, 0] = tmp_comb
        else:
            parents_comb = np.array([[]], dtype=np.int)
        #print("PARENTS COMB ", parents_comb)
        M = np.zeros([max(1, parents_comb.shape[0]), \
                      self.variables.loc[child_id, "Value"], \
                      self.variables.loc[child_id, "Value"]], dtype=np.int)
        #print(M)

        T = np.zeros([max(1, parents_comb.shape[0]), \
                      self.variables.loc[child_id, "Value"]], dtype=np.float)
        #print(T)
        #print("T Vector")
        #print(child_id)
        T_vector = np.array([self.variables.iloc[child_id, 1].astype(np.int)])
        #print(T_vector)
        #for x in parents_id:
            #print(self.variables.iloc[x, 1])
        T_vector = np.append(T_vector, [self.variables.iloc[x, 1] for x in parents_id])
        #print(T_vector)
        T_vector = T_vector.cumprod().astype(np.int)
        #print(T_vector)

        #print("M Vector")
        M_vector = np.array([self.variables.iloc[child_id, 1], self.variables.iloc[child_id, 1].astype(np.int)])
        #print(M_vector)
        M_vector = np.append(M_vector, [self.variables.iloc[x, 1] for x in parents_id])

        #for x in parents_id:
            #print(self.variables.iloc[x, 1])
        M_vector = M_vector.cumprod().astype(np.int)
        #print("MVECTOR", M_vector)

        CIM = self._compute_cim(self.trajectories, child_id, parents_id, T_vector, M_vector, parents_comb, M, T)
        return parents_comb, M, T, CIM

    def independence_test(self, to_var, from_var, sep_set, alpha_exp, alpha_chi2, thumb_threshold):
        #print("To var", to_var)
        #print("From var", from_var)
        #print("sep set", sep_set)
        parents = np.array(sep_set)
        parents = np.append(parents, from_var)
        parents.sort()
        #print("PARENTS", parents)
        parents_no_from_mask = parents != from_var
        #print("Parents Comb NO Mask", parents_no_from_mask)


        parents_comb_from, M_from, T_from, CIM_from = self.compute_cim(to_var, parents)
        #print("Parents Comb From", parents_comb_from)

        #print("C2:", CIM_from)

        if self.variables.loc[to_var, "Value"] > 2:
            df = (self.variables.loc[to_var, "Value"] - 1) ** 2
            df = df * (self.variables.loc[from_var, "Value"])
            for v in sep_set:
                df = df * (self.variables.loc[v, "Value"])

            if np.all(np.sum(np.diagonal(M_from, axis1=1, axis2=2), axis=1) / df < thumb_threshold):
                return False
            #print("Before CHi quantile", self.variables.loc[to_var, "Value"] - 1)
            chi_2_quantile = chi2_dist.ppf(1 - alpha_chi2, self.variables.loc[to_var, "Value"] - 1)
            #print("Chi Quantile", chi_2_quantile)

        parents_comb, M, T, CIM = self.compute_cim(to_var, parents[parents_no_from_mask])

        #print("C1", CIM)


        for comb_id in range(parents_comb.shape[0]):
            # Bad code, inefficient
            #print("COMB ID", comb_id)

            if parents.shape[0] > 1:
                #print("STEP 0", parents_comb_from[:, parents_no_from_mask])
                #print("STEP 1", np.all(parents_comb_from[:, parents_no_from_mask] == parents_comb[comb_id], axis=1))
                #print("STEP 2", np.argwhere(
                    #np.all(parents_comb_from[:, parents_no_from_mask] == parents_comb[comb_id], axis=1)).ravel())
                tmp_parents_comb_from_ids = np.argwhere(
                    np.all(parents_comb_from[:, parents_no_from_mask] == parents_comb[comb_id], axis=1)).ravel()
            else:
                tmp_parents_comb_from_ids = np.array([x for x in range(parents_comb_from.shape[0])])

            #print("TMP PAR COMB IDSSSS:", tmp_parents_comb_from_ids)
            for comb_from_id in tmp_parents_comb_from_ids:
                #print("COMB ID FROM", comb_from_id)
                diag = np.diag(CIM[comb_id])
                diag_from = np.diag(CIM_from[comb_from_id])
                #print("Diag C2", diag_from)
                #print("Diag C1", diag)
                r1 = np.diag(M[comb_id])
                r2 = np.diag(M_from[comb_from_id])
                stats = diag_from / diag
                #print("Exponential Test", stats, r1, r2)
                for id_diag in range(diag.shape[0]):
                    if stats[id_diag] < f_dist.ppf(alpha_exp / 2, r1[id_diag], r2[id_diag]) or \
                            stats[id_diag] > f_dist.ppf(1 - alpha_exp / 2, r1[id_diag], r2[id_diag]):
                        return False

                if diag.shape[0] > 2:

                    # https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/chi2samp.htm
                    K_from = np.sqrt(M[comb_id].diagonal() / M_from[comb_from_id].diagonal())
                    K = np.sqrt(M_from[comb_from_id].diagonal() / M[comb_id].diagonal())
                    #print("K From", K_from)
                    #print("K ", K)

                    M_no_diag = M[comb_id][~np.eye(diag.shape[0], dtype=np.bool)].reshape(diag.shape[0], -1)
                    M_from_no_diag = M_from[comb_from_id][~np.eye(diag.shape[0], dtype=np.bool)].reshape(diag.shape[0],
                                                                                                         -1)
                    #print("M No Diag", M_no_diag)
                    #print("M From No Diag", M_from_no_diag)
                    chi_stats = np.sum((np.power((M_no_diag.T * K).T - (M_from_no_diag.T * K_from).T, 2) \
                                        / (M_no_diag + M_from_no_diag)), axis=1)
                    #print("Chi stats", chi_stats)
                    #print("Chi Quantile", chi_2_quantile)
                    if np.any(chi_stats > chi_2_quantile):
                        return False

        return True

    def cb_structure_algo(self, alpha_exp=0.1, alpha_chi2=0.1, thumb_threshold=25):
        adj_matrix = np.ones((self.variables.shape[0], self.variables.shape[0]), dtype=np.bool)
        np.fill_diagonal(adj_matrix, False)
        for to_var in tqdm(range(self.variables.shape[0])):
            n = 0
            tested_variables = np.argwhere(adj_matrix[:, to_var]).ravel()
            while n < tested_variables.shape[0]:
                for from_var in tested_variables:
                    if from_var not in tested_variables:
                        continue
                    if n >= tested_variables.shape[0]:
                        break
                    sep_set_vars = tested_variables[tested_variables != from_var]
                    for comb in combinations(sep_set_vars, n):
                        if self.independence_test(to_var, from_var, comb, alpha_exp, alpha_chi2, thumb_threshold):
                            #print("######REMOVING EDGE #############", from_var, to_var)
                            adj_matrix[from_var, to_var] = False
                            tested_variables = np.argwhere(adj_matrix[:, to_var]).ravel()
                            break
                n += 1
        #print("MATRIZ:", adj_matrix)
        self.matrix =  adj_matrix


