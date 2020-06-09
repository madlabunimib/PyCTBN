import numpy as np
import sample_path as sp
import node


class DynamicGraph():

    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.graph = {}
        self.states_number = 0

    def initialize_graph(self, trajectory):
        for(node_id,state) in enumerate(trajectory.get_states()):
            n = node.Node(state, node_id)
            self.graph[state] = {"Arcs":{}, "Time":0.0, "Node":n}
            self.states_number += 1

    
    def build_graph_from_first_traj(self, trajectory):
        self.initialize_graph(trajectory)
        matrix_traj = trajectory.get_trajectory_as_matrix()
        for indx in range(len(matrix_traj) - 1):
            self.add_transaction(matrix_traj, indx)

    def add_transaction(self, matrix_traj, indx):
        current_state = matrix_traj[indx][1]
        next_state = matrix_traj[indx + 1][1]
        self.graph[current_state]["Time"] += matrix_traj[indx + 1][0] - matrix_traj[indx][0]
        next_node = self.graph[next_state]["Node"]
        if next_node not in self.graph[current_state]["Arcs"].keys():
            self.graph[current_state]["Arcs"][next_node] = 1
        else:
            self.graph[current_state]["Arcs"][next_node] += 1
    
    def append_new_trajectory(self, trajectory):
        matrix_traj = trajectory.get_trajectory_as_matrix()
        for indx in range(len(matrix_traj) - 1):
            current_state = matrix_traj[indx][1]
            next_state = matrix_traj[indx + 1][1]
            if current_state not in self.graph.keys():
                current_node = node.Node(current_state, self.states_number)
                self.graph[current_state] = {"Arcs":{}, "Time":0.0, "Node":current_node}
                self.states_number += 1
            if next_state not in self.graph.keys():
                next_node = node.Node(next_state, self.states_number)
                self.graph[next_state] = {"Arcs":{}, "Time":0.0, "Node":next_node}
                self.states_number += 1
            self.add_transaction(matrix_traj, indx)

    def build_graph(self):
        for indx, trajectory in enumerate(self.sample_path.trajectories):
            if indx == 0:
                self.build_graph_from_first_traj(trajectory)
            else:
                self.append_new_trajectory(trajectory)


######Veloci Tests#######

s1 = sp.SamplePath()
s1.build_trajectories()
print(s1.get_number_trajectories())

g1 = DynamicGraph(s1)
g1.build_graph()
print(g1.graph)
print(g1.states_number)