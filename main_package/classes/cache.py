import typing
import set_of_cims as sofc

class Cache:

    def __init__(self):
        self.list_of_sets_of_indxs = []
        self.actual_cache = []

    def find(self, parents_comb: typing.Set):
            try:
                result = self.actual_cache[self.list_of_sets_of_indxs.index(parents_comb)]
                print("CACHE HIT!!!!")
                return result
            except ValueError:
                return None

    def put(self, parents_comb: typing.Set, socim: sofc.SetOfCims):
        self.list_of_sets_of_indxs.append(parents_comb)
        self.actual_cache.append(socim)

    def clear(self):
        del self.list_of_sets_of_indxs[:]
        del self.actual_cache[:]