import os
import glob
import pandas as pd
import numpy as np




class Importer():
    """Importer forisce tutti i metodi per importare i dataset in input in pandas data_frame ed effettuare operazioni
        volte ad ottenere i valori contenuti in tali frame nel formato utile alle computazioni sui dati.
    
    :files_path: il path alla cartella contenente i dataset da utilizzare
    :df_list: lista contentente tutti i padas_data_frame che saranno importati
    """
    def __init__(self, files_path):
        self.files_path = files_path
        self.df_list = []
        #self.trajectories = []

    def import_data_from_csv(self):
        """Importa tutti i file csv presenti nel path files_path in data_frame distinti.
            Aggiunge ogni data_frame alla lista df_list.

    Parameters:
        void

    Returns:
        void

   """
        read_files = glob.glob(os.path.join(self.files_path, "*.csv"))
        for file in read_files:
            my_df = pd.read_csv(file) #TODO:Aggiungere try-catch controllo correttezza dei tipi di dato presenti nel dataset e.g. i tipi di dato della seconda colonna devono essere float
            self.df_list.append(my_df)

    
    def merge_value_columns(self, df):
        """ Effettua il merging di tutte le colonne che contengono i valori delle variabili in un unica colonna chiamata State.

        Parameters:
            df: il data_frame su cui effettuare il merging delle colonne 
        Returns:
            void
        """
        df['State'] = df[df.columns[2:]].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

    def merge_value_columns_in_all_frames(self):
        for data_frame in self.df_list:
            self.merge_value_columns(data_frame)
    
    def get_data_frames(self):
        return self.df_list

    def clear_data_frames(self):
        for data_frame in self.df_list:
            data_frame = data_frame.iloc[0:0]
    

    
    




    """def build_trajectories(self):
        for data_frame in self.df_list:
            self.merge_value_columns(data_frame)
            trajectory = data_frame[['Time','State']].to_numpy()
            self.trajectories.append(trajectory)
            #Clear the data_frame
            data_frame = data_frame.iloc[0:0]"""





#imp = Importer("../data")
#imp.import_data_from_csv()
#imp.build_trajectories()
#print(imp.trajectories[0])





