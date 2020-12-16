PyCTBN
======

A Continuous Time Bayesian Networks Library

Installation/Usage:
*******************
Download the release in .tar.gz or .whl format and simply use pip install to install it::

    $ pip install PyCTBN-1.0.tar.gz


Implementing your own data importer
***********************************
.. code-block:: python

    """This example demonstrates the implementation of a simple data importer the extends the class abstract importer to import data in csv format.
    The net in exam has three ternary nodes and no prior net structure.
    """

    from PyCTBN import AbstractImporter

    class CSVImporter(AbstractImporter):

        def __init__(self, file_path):
            self._df_samples_list = None
            super(CSVImporter, self).__init__(file_path)

        def import_data(self):
            self.read_csv_file()
            self._sorter = self.build_sorter(self._df_samples_list[0])
            self.import_variables()
            self.compute_row_delta_in_all_samples_frames(self._df_samples_list)

        def read_csv_file(self):
            df = pd.read_csv(self._file_path)
            df.drop(df.columns[[0]], axis=1, inplace=True)
            self._df_samples_list = [df]

        def import_variables(self):
            values_list = [3 for var in self._sorter]
            # initialize dict of lists
            data = {'Name':self._sorter, 'Value':values_list}
            # Create the pandas DataFrame
            self._df_variables = pd.DataFrame(data)

        def build_sorter(self, sample_frame: pd.DataFrame) -> typing.List:
            return list(sample_frame.columns)[1:]

        def dataset_id(self) -> object:
            pass

Parameters Estimation Example
*****************************

.. code-block:: python

    from PyCTBN import JsonImporter
    from PyCTBN import SamplePath
    from PyCTBN import NetworkGraph
    from PyCTBN import ParametersEstimator


    def main():
        read_files = glob.glob(os.path.join('./data', "*.json")) #Take all json files in this dir
        #import data
        importer = JsonImporter(read_files[0], 'samples', 'dyn.str', 'variables', 'Time', 'Name')
        importer.import_data(0)
        #Create a SamplePath Obj passing an already filled AbstractImporter object
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

Structure Estimation Example
****************************

.. code-block:: python

    from PyCTBN import JsonImporter
    from PyCTBN import SamplePath
    from PyCTBNimport StructureEstimator
    
    def structure_estimation_example():

        # read the json files in ./data path
        read_files = glob.glob(os.path.join('./data', "*.json"))
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