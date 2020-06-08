import pandas as pd
import numpy as np
import importer as imp
import trajectory as tr


class SamplePath():

    def __init__(self, files_path):
        self.importer = imp.Importer(files_path)
        self.trajectories = []


    def build_trajectories(self):
        self.importer.import_data_from_csv()
        self.importer.merge_value_columns_in_all_frames()
        for data_frame in self.importer.get_data_frames():
            trajectory = tr.Trajectory(data_frame)
            self.trajectories.append(trajectory)
        self.importer.clear_data_frames()

    def get_number_trajectories(self):
        return len(self.trajectories)



######Veloci Tests#######

s1 = SamplePath("../data")
s1.build_trajectories()
print(s1.get_number_trajectories())
print(type(s1.trajectories[0].get_trajectory()[0][1]))


