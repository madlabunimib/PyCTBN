import os
import glob
import pandas as pd
import json
from abstract_importer import AbstractImporter
from line_profiler import LineProfiler


class JsonImporter(AbstractImporter):
    """
    Implementa l'interfaccia AbstractImporter e aggiunge i metodi necessari a costruire le trajectories e la struttura della rete
    del dataset in formato json con la seguente struttura:
    [] 0
        |_ dyn.cims
        |_ dyn.str
        |_ samples
        |_ variabels

    :df_samples_list: lista di dataframe, ogni dataframe contiene una traj
    :df_structure: dataframe contenente la struttura della rete
    :df_variables: dataframe contenente le infromazioni sulle variabili della rete

    """

    def __init__(self, files_path, samples_label, structure_label, variables_label, time_key, variables_key):
        self.samples_label = samples_label
        self.structure_label = structure_label
        self.variables_label = variables_label
        self.time_key = time_key
        self.variables_key = variables_key
        self.df_samples_list = []
        self._df_structure = pd.DataFrame()
        self._df_variables = pd.DataFrame()
        self._concatenated_samples = None
        self.sorter = None
        super(JsonImporter, self).__init__(files_path)

    def import_data(self):
        raw_data = self.read_json_file()
        self.import_trajectories(raw_data)
        self.compute_row_delta_in_all_samples_frames(self.time_key)
        self.clear_data_frame_list()
        self.import_structure(raw_data)
        self.import_variables(raw_data, self.sorter)

    def import_trajectories(self, raw_data):
        self.normalize_trajectories(raw_data, 0, self.samples_label)

    def import_structure(self, raw_data):
        self._df_structure = self.one_level_normalizing(raw_data, 0, self.structure_label)

    def import_variables(self, raw_data, sorter):
        self._df_variables = self.one_level_normalizing(raw_data, 0, self.variables_label)
        self._df_variables[self.variables_key] = self._df_variables[self.variables_key].astype("category")
        self._df_variables[self.variables_key] = self._df_variables[self.variables_key].cat.set_categories(sorter)
        self._df_variables = self._df_variables.sort_values([self.variables_key])

    def read_json_file(self):
        """
        Legge il primo file .json nel path self.filepath

        Parameters:
              void
        Returns:
              :data: il contenuto del file json

        """
        try:
            read_files = glob.glob(os.path.join(self.files_path, "*.json"))
            if not read_files:
                raise ValueError('No .json file found in the entered path!')
            with open(read_files[0]) as f:
                data = json.load(f)
                return data
        except ValueError as err:
            print(err.args)

    def one_level_normalizing(self, raw_data, indx, key):
        """
        Estrae i dati innestati di un livello, presenti nel dataset raw_data,
        presenti nel json array all'indice indx nel json object key

        Parameters:
            :raw_data: il dataset json completo
            :indx: l'indice del json array da cui estrarre i dati
            :key: il json object da cui estrarre i dati
        Returns:
            Il dataframe contenente i dati normalizzati

        """
        return pd.DataFrame(raw_data[indx][key])

    def normalize_trajectories(self, raw_data, indx, trajectories_key):
        """
        Estrae le traiettorie presenti in rawdata nel json array all'indice indx, nel json object trajectories_key.
        Aggiunge le traj estratte nella lista di dataframe self.df_samples_list

        Parameters:
            void
        Returns:
            void
        """
        for sample_indx, sample in enumerate(raw_data[indx][trajectories_key]):
            self.df_samples_list.append(pd.DataFrame(sample))

    def compute_row_delta_sigle_samples_frame(self, sample_frame, time_header_label, columns_header, shifted_cols_header):
        sample_frame[time_header_label] = sample_frame[time_header_label].diff().shift(-1)
        shifted_cols = sample_frame[columns_header[1:]].shift(-1)
        shifted_cols.columns = shifted_cols_header
        sample_frame = sample_frame.assign(**shifted_cols)
        sample_frame.drop(sample_frame.tail(1).index, inplace=True)
        return sample_frame

    def compute_row_delta_in_all_samples_frames(self, time_header_label):
        columns_header = list(self.df_samples_list[0].columns.values)
        self.sorter = columns_header[1:]
        shifted_cols_header = [s + "S" for s in columns_header[1:]]
        for indx, sample in enumerate(self.df_samples_list):
            self.df_samples_list[indx] = self.compute_row_delta_sigle_samples_frame(sample,
                                                        time_header_label, columns_header, shifted_cols_header)
            #print(self.df_samples_list[indx])
        self._concatenated_samples = pd.concat(self.df_samples_list)

    def build_list_of_samples_array(self, data_frame):
        """
        Costruisce una lista contenente le colonne presenti nel dataframe data_frame convertendole in numpy_array
        Parameters:
            :data_frame: il dataframe da cui estrarre e convertire le colonne
        Returns:
            :columns_list: la lista contenente le colonne convertite in numpyarray

        """
        columns_list = []
        for column in data_frame:
            columns_list.append(data_frame[column].to_numpy())
        return columns_list

    def clear_data_frames(self):
        """
        Rimuove tutti i valori contenuti nei data_frames presenti in df_samples_list
        Parameters:
            void
        Returns:
            void
         """
        self._concatenated_samples = self._concatenated_samples.iloc[0:0]

    def clear_data_frame_list(self):
        for indx in range(len(self.df_samples_list)):  # Le singole traj non servono più
            self.df_samples_list[indx] = self.df_samples_list[indx].iloc[0:0]

    @property
    def concatenated_samples(self):
        return self._concatenated_samples

    @property
    def variables(self):
        return self._df_variables

    @property
    def structure(self):
        return self._df_structure




