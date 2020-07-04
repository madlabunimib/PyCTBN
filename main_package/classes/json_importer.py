import os
import glob
import pandas as pd
import json
from abstract_importer import AbstractImporter


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

    def __init__(self, files_path):
        self.df_samples_list = []
        self.concatenated_samples = None
        self.df_structure = pd.DataFrame()
        self.df_variables = pd.DataFrame()
        super(JsonImporter, self).__init__(files_path)

    def import_data(self):
        raw_data = self.read_json_file()
        self.import_trajectories(raw_data)
        self.import_structure(raw_data)
        self.import_variables(raw_data)

    def import_trajectories(self, raw_data):
        self.normalize_trajectories(raw_data, 0, 'samples')

    def import_structure(self, raw_data):
        self.df_structure = self.one_level_normalizing(raw_data, 0, 'dyn.str')

    def import_variables(self, raw_data):
        self.df_variables = self.one_level_normalizing(raw_data, 0, 'variables')

    def read_json_file(self):
        """
        Legge 'tutti' i file .json presenti nel path self.filepath

        Parameters:
              void
        Returns:
              :data: il contenuto del file json

        """
        try:
            read_files = glob.glob(os.path.join(self.files_path, "*.json"))
            for file_name in read_files:
                with open(file_name) as f:
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
        return pd.json_normalize(raw_data[indx][key])

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
            self.df_samples_list.append(pd.json_normalize(raw_data[indx][trajectories_key][sample_indx]))

    def compute_row_delta_sigle_samples_frame(self, sample_frame):
        columns_header = list(sample_frame.columns.values)
        #print(columns_header)
        for col_name in columns_header:
            if col_name == 'Time':
                sample_frame[col_name + 'Delta'] = sample_frame[col_name].diff()
            #else:
                #sample_frame[col_name + 'Delta'] = (sample_frame[col_name].diff().bfill() != 0).astype(int)
        #sample_frame['Delta'] = sample_frame['Time'].diff()
        #print(sample_frame)

    def compute_row_delta_in_all_samples_frames(self):
        for sample in self.df_samples_list:
            self.compute_row_delta_sigle_samples_frame(sample)
        self.concatenated_samples = pd.concat(self.df_samples_list)
        self.concatenated_samples['Time'] = self.concatenated_samples['TimeDelta']
        del self.concatenated_samples['TimeDelta']
        self.concatenated_samples['Time'] = self.concatenated_samples['Time'].fillna(0)


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
        for indx in range(len(self.df_samples_list)):
            self.df_samples_list[indx] = self.df_samples_list[indx].iloc[0:0]
        self.concatenated_samples = self.concatenated_samples.iloc[0:0]


"""ij = JsonImporter("../data")
ij.import_data()
#print(ij.df_samples_list[7])
print(ij.df_structure)
print(ij.df_variables)
#print((ij.build_list_of_samples_array(0)[1].size))
#ij.compute_row_delta_sigle_samples_frame(ij.df_samples_list[0])
ij.compute_row_delta_in_all_samples_frames()
print(ij.concatenated_samples.to_numpy())"""
