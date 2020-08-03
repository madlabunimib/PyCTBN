import typing
import set_of_cims as sofc


class Cache:

    def __init__(self):
        self.list_of_sets_of_indxs = []
        self.actual_cache = []

    def find(self, parents_comb: typing.Union[typing.Set, str]):
            try:
                #print("Cache State:", self.list_of_sets_of_indxs)
                #print("Look For:", parents_comb)
                result = self.actual_cache[self.list_of_sets_of_indxs.index(parents_comb)]
                print("CACHE HIT!!!!", parents_comb)
                return result
            except ValueError:
                return None

    def put(self, parents_comb: typing.Union[typing.Set, str], socim: sofc.SetOfCims):
        #print("Putting in cache:", parents_comb)
        self.list_of_sets_of_indxs.append(parents_comb)
        self.actual_cache.append(socim)

    def clear(self):
        del self.list_of_sets_of_indxs[:]
        del self.actual_cache[:]