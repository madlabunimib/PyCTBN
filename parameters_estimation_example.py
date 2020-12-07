
import os
import glob


from PyCTBN.PyCTBN.json_importer import JsonImporter
from PyCTBN.PyCTBN.sample_path import SamplePath
from PyCTBN.PyCTBN.network_graph import NetworkGraph
from PyCTBN.PyCTBN.parameters_estimator import ParametersEstimator


def main():
    read_files = glob.glob(os.path.join('./data', "*.json")) #Take all json files in this dir
    #import data
    importer = JsonImporter(read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
    importer.import_data(0)
    #Create a SamplePath Obj
    s1 = SamplePath(importer)
    #Build The trajectries and the structural infos
    s1.build_trajectories()
    s1.build_structure()
    print(s1.structure.edges)
    print(s1.structure.nodes_values)
    #From The Structure Object build the Graph
    g = NetworkGraph(s1.structure)
    #Select a node you want to estimate the parameters
    node = g.nodes[2]
    print("Node", node)
    #Init the _graph specifically for THIS node
    g.fast_init(node)
    #Use SamplePath and Grpah to create a ParametersEstimator Object
    p1 = ParametersEstimator(s1.trajectories, g)
    #Init the peEst specifically for THIS node
    p1.fast_init(node)
    #Compute the parameters
    sofc1 = p1.compute_parameters_for_node(node)
    #The est CIMS are inside the resultant SetOfCIms Obj
    print(sofc1.actual_cims)


if __name__ == "__main__":
    main()
