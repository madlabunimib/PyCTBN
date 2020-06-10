import numpy as np
import sample_path as sp
import node


class DynamicGraph():
    """
    Rappresenta un grafo dinamico con la seguente struttura:
        {Key: {Arcs:{node_object: #in_arcs.....}, Time:t, Node:node_object}.....}
        Key = lo state_id del nodo
        Arcs = la lista di adiacenza del nodo identificato dalla stringa state_id, contenente oggetti di tipo Node
        Time = lo state residence time del nodo identificato dalla stringa state_id
        Node = L'oggetto Node con lo stesso state_id  pari a Key e node_id opportunamento inizalizzato

        :sample_path: le traiettorie/a da cui costruire il grafo
        :graph: la struttura dinamica che definisce il grafo
        :states_number: il numero di nodi contenuti nel grafo graph
    """

    def __init__(self, sample_path):
        self.sample_path = sample_path
        self.graph = {}
        self.states_number = 0

    def initialize_graph(self, trajectory):
        """
        Data la traiettoria trajectory inizializza le chiavi del dizionario graph e crea le opportune sottostrutture interne.

        """
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
        """
            Aggiunge la transizione dal nodo current_state al nodo next_state e aggiorna lo state residence time del nodo
            con chiave current_state.

            Parameters:
                matrix_traj: la traiettoria con cui si sta costruendo il grafo graph
                indx: la posizione nella visita
            Returns:
                void
        """
        current_state = matrix_traj[indx][1]
        next_state = matrix_traj[indx + 1][1]
        self.graph[current_state]["Time"] += matrix_traj[indx + 1][0] - matrix_traj[indx][0]
        next_node = self.graph[next_state]["Node"] #Punta all'oggeto Node con node_id precedentemente settato
        if next_node not in self.graph[current_state]["Arcs"].keys():  #Se non hai ancora incontrato next_node inizializza il numero di archi entranti
            self.graph[current_state]["Arcs"][next_node] = 1
        else:
            self.graph[current_state]["Arcs"][next_node] += 1
    
    def append_new_trajectory(self, trajectory):
        """
        Aggiunge i risultati di una nuova esplorazione trajectory al grafo graph.

        Parameters:
            trajectory: la traiettoria da aggiungere
        Returns:
            void
        """
        matrix_traj = trajectory.get_trajectory_as_matrix()
        
        current_state = matrix_traj[0][1] #Aggiungi se necessario i primi due stati
        next_state = matrix_traj[1][1]
        self.add_node_if_not_present(current_state) 
        self.add_node_if_not_present(next_state) 
        self.add_transaction(matrix_traj, 0)
        
        for indx in range(1, (len(matrix_traj) - 1)):
            current_state = matrix_traj[indx][1]
            next_state = matrix_traj[indx + 1][1]
            self.add_node_if_not_present(next_state) 
            self.add_transaction(matrix_traj, indx)

    def add_node_if_not_present(self, current_state):
        if current_state not in self.graph.keys(): #Se non hai ancora incontrato il lo state current_state
            current_node = node.Node(current_state, self.states_number) #Crea il l'oggetto con opportuno node_id
            self.graph[current_state] = {"Arcs":{}, "Time":0.0, "Node":current_node} #Aggiungilo al dizionario graph
            self.states_number += 1

    def build_graph(self):
        for indx, trajectory in enumerate(self.sample_path.trajectories):
            if indx == 0:
                self.build_graph_from_first_traj(trajectory)
            else:
                self.append_new_trajectory(trajectory)

    def get_root_node(self):
        return self.graph[list(self.graph)[0]]["Node"]

    def get_neighbours(self, node):
        return self.graph[node.state_id]["Arcs"]


######Veloci Tests#######

#s1 = sp.SamplePath()
#s1.build_trajectories()
#print(s1.get_number_trajectories())

#g1 = DynamicGraph(s1)
#g1.build_graph()
#print(g1.graph)
#print(g1.states_number)