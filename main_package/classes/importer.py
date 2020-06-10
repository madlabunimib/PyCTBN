import os
import glob
import pandas as pd
import numpy as np




class Importer():
    """
    Fornisce tutti i metodi per importare i dataset contenuti nella directory files_path 
    in pandas data_frame. 
    Permette di effettuare operazioni
    volte ad ottenere i valori contenuti in tali frame nel formato utile alle computazioni successive.
    
    :files_path: il path alla cartella contenente i dataset da utilizzare
    :df_list: lista contentente tutti i padas_data_frame contenuti nella directory files_path
    """
    def __init__(self, files_path):
        self.files_path = files_path
        self.df_list = []
        

    def import_data_from_csv(self):
        """
        Importa tutti i file csv presenti nel path files_path in data_frame distinti.
        Aggiunge ogni data_frame alla lista df_list.
        Parameters:
            void
        Returns:
            void
         """
        try:
            read_files = glob.glob(os.path.join(self.files_path, "*.csv"))
            for file in read_files:
                my_df = pd.read_csv(file)
                self.check_types_validity(my_df)
                self.df_list.append(my_df)
        except ValueError as err:
            print(err.args)

        

    def merge_value_columns(self, df):
        """ 
        Effettua il merging di tutte le colonne che contengono i valori delle variabili 
        in un unica colonna chiamata State.
        Parameters:
            df: il data_frame su cui effettuare il merging delle colonne 
        Returns:
            void
        """
        df['State'] = df[df.columns[2:]].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

    def merge_value_columns_in_all_frames(self):
        """
        Richiama il metodo  merge_value_columns su tutti i data_frame contenuti nella df_list
        Parameters:
            void
        Returns:
            void

         """
        for data_frame in self.df_list:
            self.merge_value_columns(data_frame)

    def drop_unneccessary_columns(self, df):
        """
        Rimuove le colonne contenenti i valori delle variabili dal data_frame df.
        Parameters:
            df: il data_frame su cui rimuovere le colonne
        Returns:
            void

         """
        cols = df.columns.values[2:-1]
        df.drop(cols, axis=1, inplace=True)

    def drop_unneccessary_columns_in_all_frames(self):
        """
        Richiama il metodo drop_unneccessary_columns su tutti i data_frame contenuti nella df_list

        Parameters:
            void
        Returns:
            void

         """
        for data_frame in self.df_list:
            self.drop_unneccessary_columns(data_frame)

    def get_data_frames(self):
        """
        Restituisce la lista contenente tutti i data_frames caricati in df_list.
        Parameters:
            void
        Returns:
            data_frames list
         """
        return self.df_list

    def clear_data_frames(self):
        """
        Rimuove tutti i valori contenuti nei data_frames presenti in df_list
        Parameters:
            void
        Returns:
            void
         """
        for data_frame in self.df_list:
            data_frame = data_frame.iloc[0:0]


    def check_types_validity(self, data_frame):  #Solo un esempio di controllo sui valori contenuti nel dataset
        """
        Controlla la correttezza dei tipi contenuti nei dati caricati nel data_frame (in questo caso solo che nella seconda colonna siano contenuti dei float)
         """
        if data_frame.iloc[:,1].dtype != np.float64:
            raise ValueError("The Dataset is Not in the correct format")

        #TODO Integrare i controlli di correttezza sul dataset e capire quali assunzioni vanno fatte







