

class PriorityQueue():
    """
    Rappresenta una semplice coda con priorità FIFO
    :queue: lista che contiene i valori della coda
    """

    def __init__(self):
        self.queue = []

    def enqueue(self, node):
        self.queue.append(node)

    def dequeue(self):
        return self.queue.pop(0)

    def is_empty(self):
        return len(self.queue) == 0
