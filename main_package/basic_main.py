import glob
import os

import sys
sys.path.append("./classes/")

import network_graph as ng
import sample_path as sp
import parameters_estimator as pe
import json_importer as ji


def main():
    read_files = glob.glob(os.path.join('./data', "*.json")) #Take all json files in this dir
    #import data
    importer = ji.JsonImporter(read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name', 1)
    #Create a SamplePath Obj
    s1 = sp.SamplePath(importer)
    #Build The trajectries and the structural infos
    s1.build_trajectories()
    s1.build_structure()
    print(s1.structure.edges)
    print(s1.structure.nodes_values)
    #From The Structure Object build the Graph
    g = ng.NetworkGraph(s1.structure)
    #Select a node you want to estimate the parameters
    node = g.nodes[1]
    print("NOde", node)
    #Init the _graph specifically for THIS node
    g.fast_init(node)
    #Use SamplePath and Grpah to create a ParametersEstimator Object
    p1 = pe.ParametersEstimator(s1.trajectories, g)
    #Init the peEst specifically for THIS node
    p1.fast_init(node)
    #Compute the parameters
    sofc1 = p1.compute_parameters_for_node(node)
    #The est CIMS are inside the resultant SetOfCIms Obj
    print(sofc1.actual_cims)

if __name__ == "__main__":
    main()
