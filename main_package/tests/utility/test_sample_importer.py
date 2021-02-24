
import unittest
import os
import glob
import numpy as np
import pandas as pd
from ...classes.utility.sample_importer import SampleImporter
from ...classes.structure_graph.sample_path import SamplePath

import json



class TestSampleImporter(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        with open("./main_package/data/networks_and_trajectories_binary_data_01_3.json") as f:
            raw_data = json.load(f)

            trajectory_list_raw= raw_data[0]["samples"]

            cls.trajectory_list = [pd.DataFrame(sample) for sample in trajectory_list_raw]

            cls.variables= pd.DataFrame(raw_data[0]["variables"])
            cls.prior_net_structure = pd.DataFrame(raw_data[0]["dyn.str"])


    def test_init(self):
        sample_importer = SampleImporter(
                                        trajectory_list=self.trajectory_list,
                                        variables=self.variables,
                                        prior_net_structure=self.prior_net_structure
                                    )
        
        sample_importer.import_data()

        s1 = SamplePath(sample_importer)
        s1.build_trajectories()
        s1.build_structure()
        s1.clear_memory() 

        self.assertEqual(len(s1._importer._df_samples_list), 300)
        self.assertIsInstance(s1._importer._df_samples_list,list)
        self.assertIsInstance(s1._importer._df_samples_list[0],pd.DataFrame)
        self.assertEqual(len(s1._importer._df_variables), 3)
        self.assertIsInstance(s1._importer._df_variables,pd.DataFrame)
        self.assertEqual(len(s1._importer._df_structure), 2)
        self.assertIsInstance(s1._importer._df_structure,pd.DataFrame)

    def test_order(self):
        sample_importer = SampleImporter(
                                        trajectory_list=self.trajectory_list,
                                        variables=self.variables,
                                        prior_net_structure=self.prior_net_structure
                                    )
        
        sample_importer.import_data()

        s1 = SamplePath(sample_importer)
        s1.build_trajectories()
        s1.build_structure()
        s1.clear_memory() 

        for count,var in enumerate(s1._importer._df_samples_list[0].columns[1:]):
            self.assertEqual(s1._importer._sorter[count],var)



    def ordered(self, obj):
        if isinstance(obj, dict):
            return sorted((k, self.ordered(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return sorted(self.ordered(x) for x in obj)
        else:
            return obj


if __name__ == '__main__':
    unittest.main()
