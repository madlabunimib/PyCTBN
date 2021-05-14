from PyCTBN.PyCTBN.structure_graph.trajectory_generator import TrajectoryGenerator
from PyCTBN.PyCTBN.structure_graph.network_generator import NetworkGenerator
from PyCTBN.PyCTBN.utility.json_importer import JsonImporter
from PyCTBN.PyCTBN.utility.json_exporter import JsonExporter
from PyCTBN.PyCTBN.structure_graph.structure import Structure
from PyCTBN.PyCTBN.structure_graph.sample_path import SamplePath
from PyCTBN.PyCTBN.estimators.structure_constraint_based_estimator import StructureConstraintBasedEstimator

""" 
    if __name__ == "__main__":
        trajectories = tg.multi_trajectory(t_ends = [100, 100, 100])
"""

# Network Generation
labels = ["X", "Y", "Z"]
card = 3
vals = [card for l in labels]
cim_min = 1
cim_max = 3
ng = NetworkGenerator(labels, vals)
ng.generate_graph(0.3)
ng.generate_cims(cim_min, cim_max)

# Trajectory Generation
print(ng.dyn_str)
e1 = JsonExporter(ng.variables, ng.dyn_str, ng.cims)
tg = TrajectoryGenerator(variables = ng.variables, dyn_str = ng.dyn_str, dyn_cims = e1.cims_to_json())
sigma = tg.CTBN_Sample(max_tr = 30000)
e1.add_trajectory(sigma)
e1.out_file("example.json")

# Network Estimation (Constraint Based)
importer = JsonImporter(file_path="example.json", samples_label='samples',
                        structure_label='dyn.str', variables_label='variables',
                        time_key='Time', variables_key='Name')
importer.import_data(0)
s1 = SamplePath(importer=importer)
s1.build_trajectories()
s1.build_structure()
se1 = StructureConstraintBasedEstimator(sample_path=s1, exp_test_alfa=0.1, chi_test_alfa=0.1,
                                        known_edges=[], thumb_threshold=25)
edges = se1.estimate_structure(True)
# se1.save_plot_estimated_structure_graph('./result1.png')
print(se1.adjacency_matrix())
print(edges)