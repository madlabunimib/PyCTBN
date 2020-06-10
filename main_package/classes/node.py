

class Node():
    """
    Astrae il concetto di nodo appartenente ad un grafo G.
    Un nodo viene univocamente identificato dal suo state_id, il tag node_id è utilizzato per stabilire la posizione
    del nodo stesso nel grafo rappresentato attraverso la matrice di adiacenza.
    :state_id: UID del nodo
    :node_id: int
    """

    def __init__(self, state_id, node_id=-1):
        self.state_id = state_id
        self.node_id = node_id

    def __key(self):
        return (self.state_id)

    def __hash__(self):
        return hash(self.__key())

    
    def __eq__(self, other):
        """
        Controlla l'uguaglianza di due oggetti Node.
        Due nodi sono uguali sse hanno lo stesso state_id
        Parameters:
            other: oggetto nodo con cui effettuare il confronto
        Returns:
            boolean
        """
        if isinstance(other, Node):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self):
        return "<%s, %d>"% (self.state_id, self.node_id)