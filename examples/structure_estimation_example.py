import os
import glob

from PyCTBN.classes.json_importer import JsonImporter
from PyCTBN.classes.sample_path import SamplePath
from PyCTBN.classes.structure_estimator import StructureEstimator


def structure_estimation_example():

    # read the json files in ./data path
    read_files = glob.glob(os.path.join('../data', "*.json"))
    # initialize a JsonImporter object for the first file
    importer = JsonImporter(read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
    # import the data at index 0 of the outer json array
    importer.import_data(0)
    # construct a SamplePath Object passing a filled AbstractImporter
    s1 = SamplePath(importer)
    # build the trajectories
    s1.build_trajectories()
    # build the real structure
    s1.build_structure()
    # construct a StructureEstimator object
    se1 = StructureEstimator(s1, 0.1, 0.1)
    # call the ctpc algorithm
    se1.ctpc_algorithm()
    # the adjacency matrix of the estimated structure
    print(se1.adjacency_matrix())
    # save results to a json file
    se1.save_results()
