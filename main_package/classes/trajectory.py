import pandas as pd
import numpy as np
import importer


class Trajectory():

    def __init__(self, data_frame):
        self.actual_trajectory = self.build_trajectory(data_frame)

    
    def build_trajectory(self, data_frame):
        return data_frame[['Time','State']].to_numpy()

    def get_trajectory(self):
        return self.actual_trajectory
