import pandas as pd
import numpy as np


class Trajectory():

    def __init__(self, data_frame):
        self.actual_trajectory = data_frame
        
    def get_trajectory(self):
        return self.actual_trajectory
    
    def get_trajectory_as_matrix(self):
        return self.actual_trajectory[['Time','State']].to_numpy()

    def get_states(self):
        return self.actual_trajectory['State'].unique()
