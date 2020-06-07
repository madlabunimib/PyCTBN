import os
import glob
import pandas as pd
import numpy as np




class Importer():
    """Importer forisce tutti i metodi per portare il dataset in input nelle strutture dati corrette per essere trattate
    in memoria...... 
    
    :files_path: il path alla cartella contenente i dataset da utilizzare
    """
    def __init__(self, files_path):
        self.files_path = files_path
        self.df_list = []
        self.trajectories = []

    def import_data_from_csv(self):
        read_files = glob.glob(os.path.join(self.files_path, "*.csv"))
        for file in read_files:
            my_df = pd.read_csv(file) #TODO:Aggiungere try-catch controllo correttezza dei tipi di dato presenti nel dataset e.g. i tipi di dato della seconda colonna devono essere float
            self.df_list.append(my_df)

    def merge_value_columns(self, df):
        df['State'] = df[df.columns[2:]].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

    def build_trajectories(self):
        for data_frame in self.df_list:
            self.merge_value_columns(data_frame)
            trajectory = data_frame[['Time','State']].to_numpy()
            self.trajectories.append(trajectory)
            #Clear the data_frame
            data_frame = data_frame.iloc[0:0]





imp = Importer("../data")
imp.import_data_from_csv()
imp.build_trajectories()
print(imp.trajectories[0])
#print(len(imp.df_list))
#print(imp.df_list[0])
#for column in imp.df_list[0].columns[2:]:
    #print(imp.df_list[0][column])
    #imp.df_list[0]['State'] = imp.df_list[0][column].astype(str) 
#imp.df_list[0]['State'] = imp.df_list[0][imp.df_list[0].columns[2:]].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

#imp.df_list[0]['new'] = imp.df_list[0].astype(str).values.sum(axis=1)
#print(imp.df_list[0])

#trajectory  =  imp.df_list[0][['Time','State']].to_numpy()
#print(hash(trajectory[0][1]))
#print(hash(trajectory[0][1]))
#imp.df_list[0] = imp.df_list[0].iloc[0:0]
#print(imp.df_list[0])

#print(type(imp.df_list[0].iloc[0,2]))
#print(imp.df_list[0].iloc[:, 2:])
#imp.merge_columns_values((imp.df_list[0]))
#print(type(imp.df_list[0].iloc[0,2]))




