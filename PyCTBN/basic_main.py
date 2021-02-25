import glob
import os

import sys
sys.path.append("./PyCTBN/")

import structure_graph.network_graph as ng
import structure_graph.sample_path as sp
import structure_graph.set_of_cims as sofc
import estimators.parameters_estimator as pe
import utility.json_importer as ji


def main():
    read_files = glob.glob(os.path.join('./data', "*.json")) #Take all json files in this dir
    #import data
    importer = ji.JsonImporter(read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
    #Create a SamplePath Obj
    s1 = sp.SamplePath(importer)
    #Build The trajectries and the structural infos
    s1.build_trajectories()
    s1.build_structure()
    #From The Structure Object build the Graph
    g = ng.NetworkGraph(s1.structure)
    #Select a node you want to estimate the parameters
    node = g.nodes[1]
    #Init the graph specifically for THIS node
    g.fast_init(node)
    #Use SamplePath and Grpah to create a ParametersEstimator Object
    p1 = pe.ParametersEstimator(s1, g)
    #Init the peEst specifically for THIS node
    p1.fast_init(node)
    #Compute the parameters
    sofc1 = p1.compute_parameters_for_node(node)
    #The est CIMS are inside the resultant SetOfCIms Obj
    print(sofc1.actual_cims)

if __name__ == "__main__":
    main()
