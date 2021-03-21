PyCTBN
======

A Continuous Time Bayesian Networks Library

Installation/Usage
*******************
Download the release in .tar.gz or .whl format and simply use pip install to install it::

    $ pip install PyCTBN-1.0.tar.gz

Documentation
*************
Please refer to https://philipmartini.github.io/PyCTBN/ for the full project documentation.

Implementing your own data importer
***********************************
| This example demonstrates the implementation of a simple data importer the extends the class AbstractImporter 
| to import data in csv format. The net in exam has three ternary nodes and no prior net structure.
| Suppose the trajectories that have to be inported have this structure:

.. image:: docs-out/esempio_dataset.png
  :width: 600
  :alt: An example trajectory to be imported.
 
| In the read_csv_file method the data are imported in memory, put in a list and  assigned to the _df_samples_list class
| member, so that it contains all the trajectories to be processed.
| In the import_variables method the dataframe containing the nodes labels and the cardinalities of the nodes 
| is assigned to the _df_variables class member. 
| The class member _sorter has to contain the nodes labels in the same order of the trajectory columns,
| just override the build_sorter method to do that.
| If your datasets names have particular id, you can keep it using the dataset_id method to assign the id to a new class member. 
| Finally the import_data method call all the previously implemented methods and calls the compute_row_delta_in_all_samples_frames
| to process all the trajectories in _df_samples_list.
| For more information about the class memebers and methods of AbstractImporter please refer to the documentation.

.. code-block:: python

    import pandas as pd
    import typing
    
    from PyCTBN import AbstractImporter
    from PyCTBN import SamplePath

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

    def main():
        # create the importer object
        csvimp = CSVImporter('/dataset_example.csv')
        # call the wrapping method that wil import and process the data
        csvimp.import_data()
        # pass the AbstractImporter object to the SamplePath constructor
        s1 = SamplePath(csvimp)
        # SamplePath will contain the Trajecotry object...
        s1.build_trajectories()
        #...and the Structure object with all the process data
        s1.build_structure()

Structure Estimation Examples
****************************
| This example shows how to estimate the structure given a series of trajectories using a constraint based approach.
| The first three instructions import all the necessary data (trajectories, nodes cardinalities, nodes labels),
| and are contextual to the dataset that is been used, in the code comments are marked as optional <>.
| If your data has a different structure or format you should implement your own importer 
| (see Implementing your own importer example).
| The other instructions are not optional and should follow the same order.
| A SamplePath object is been created, passing an AbstractImporter object that contains the  correct class members 
| filled with the data that are necessary to estimate the structure.
| Next the build_trajectories  and build_structure methods are called to instantiate the objects that will contain
| the processed trajectories and all the net information.
| Then an estimator object is created, in this case a constraint based estimator, 
| it necessary to pass a SamplePath object where build_trajectories and build_structure methods have already been called.
| If you have prior knowledge about the net structure pass it to the constructor with the known_edges parameter.
| The other three parameters are contextual to the StructureConstraintBasedEstimator, see the documentation for more details.
| To estimate the structure simply call the estimate_structure method.
| You can obtain the estimated structure as a boolean adjacency matrix with the method adjacency_matrix, 
| or save it as a json file that contains all the nodes labels, and obviously the estimated edges.
| You can also save a graphical model representation of the estimated structure 
| with the save_plot_estimated_structure_graph.

.. code-block:: python

    import glob
    import os

    from PyCTBN import JsonImporter
    from PyCTBN import SamplePath
    from PyCTBN import StructureConstraintBasedEstimator


    def structure_constraint_based_estimation_example():
        # <read the json files in ./data path>
        read_files = glob.glob(os.path.join('./data', "*.json"))
        # <initialize a JsonImporter object for the first file>
        importer = JsonImporter(file_path=read_files[0], samples_label='samples',
                                structure_label='dyn.str', variables_label='variables',
                                time_key='Time', variables_key='Name')
        # <import the data at index 0 of the outer json array>
        importer.import_data(0)
        # construct a SamplePath Object passing a filled AbstractImporter object
        s1 = SamplePath(importer=importer)
        # build the trajectories
        s1.build_trajectories()
        # build the information about the net
        s1.build_structure()
        # construct a StructureEstimator object passing a correctly build SamplePath object
        # and the independence tests significance, if you have prior knowledge about 
        # the net structure create a list of tuples
        # that contains them and pass it as known_edges parameter
        se1 = StructureConstraintBasedEstimator(sample_path=s1, exp_test_alfa=0.1, chi_test_alfa=0.1,
                                                known_edges=[], thumb_threshold=25)
        # call the algorithm to estimate the structure
        se1.estimate_structure()
        # obtain the adjacency matrix of the estimated structure
        print(se1.adjacency_matrix())
        # save the estimated structure  to a json file 
        # (remember to specify the path AND the .json extension)....
        se1.save_results('./results0.json')
        # ...or save it also in a graphical model fashion 
        # (remember to specify the path AND the .png extension)
        se1.save_plot_estimated_structure_graph('./result0.png')
