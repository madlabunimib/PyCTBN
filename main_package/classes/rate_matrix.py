import numpy as np
import dynamic_graph as dg
import sample_path as sp
import priority_queue as pq


class RateMatrix():
    """
    Rappresenta la matrice Q di una generica CTMC costruita a partire dalle informazioni contenute nel grafo dinamico
    """
    def __init__(self, graph, dim):
        self.graph = graph
        self.matrix = np.zeros(shape=(dim,dim))
        self.pr_queue = pq.PriorityQueue()

    def build_matrix(self):
        root = self.graph.get_root_node()
        root.color = dg.node.Color.GRAY
        self.pr_queue.enqueue(root)
        

        while not self.pr_queue.is_empty():
            n = self.pr_queue.dequeue()
            adjacency_list = self.graph.get_neighbours(n)
            #print(adjacency_list)
            time = self.graph.graph[n.state_id]["Time"]
            sum_of_qs = 0.0
            for nd, weight in adjacency_list.items():
                sum_of_qs += self.calculate_off_diagonal_element_and_fill_matrix(n.node_id, nd.node_id, weight, time)
                if self.graph.graph[nd.state_id]["Node"].color == dg.node.Color.WHITE:
                    self.graph.graph[nd.state_id]["Node"].color = dg.node.Color.GRAY
                    self.pr_queue.enqueue(nd)
            n.color = dg.node.Color.BLACK
            self.calculate_diagonal_element_and_fill_matrix(sum_of_qs, n.node_id)


    def calculate_off_diagonal_element_and_fill_matrix(self, start_node, arrival_node, weight, time):
        q = weight / time
        self.matrix[start_node][arrival_node] = q
        return q

    def calculate_diagonal_element_and_fill_matrix(self, sum_of_qs, start_node):
        self.matrix[start_node][start_node] = -sum_of_qs


# A Simple Test #
s1 = sp.SamplePath()
s1.build_trajectories()
print(s1.get_number_trajectories())

g1 = dg.DynamicGraph(s1)
g1.build_graph()
print(g1.graph)
#print(g1.states_number)

Q = RateMatrix(g1, g1.states_number)
#print(Q.matrix)
Q.build_matrix()
print(Q.matrix)

non_zero_values = 0
val = 0.0
for coeff in Q.matrix[0][1:]:
    if(coeff != 0):
        non_zero_values += 1
        val += coeff

print(non_zero_values == len(Q.graph.graph["222"]["Arcs"].keys()))