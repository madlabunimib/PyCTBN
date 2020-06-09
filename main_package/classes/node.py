

class Node():

    def __init__(self, state_id, node_id=-1):
        self.state_id = state_id
        self.node_id = node_id

    def __key(self):
        return (self.state_id)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self):
        return "<%s, %d>"% (self.state_id, self.node_id)