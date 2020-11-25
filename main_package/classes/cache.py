import typing

import set_of_cims as sofc


class Cache:
    """
    This class has the role of a cache for SetOfCIMS of a test node that have been already computed during the ctpc algorithm.

    :_list_of_sets_of_parents: a list of Sets of the parents to which the cim in cache at SAME index is related
    :_actual_cache: a list of setOfCims objects
    """

    def __init__(self):
        self._list_of_sets_of_parents = []
        self._actual_cache = []

    def find(self, parents_comb: typing.Set) -> sofc.SetOfCims:
        """
        Tries to find in cache given the symbolic parents combination parents_comb the SetOfCims related to that parents_comb.
        Parameters:
            parents_comb: the parents related to that SetOfCims
        Returns:
            A SetOfCims object if the parents_comb index is found in _list_of_sets_of_parents.
            None otherwise.

        """
        try:
            #print("Cache State:", self.list_of_sets_of_indxs)
            #print("Look For:", parents_comb)
            result = self._actual_cache[self._list_of_sets_of_parents.index(parents_comb)]
            print("CACHE HIT!!!!", parents_comb)
            return result
        except ValueError:
            return None

    def put(self, parents_comb: typing.Union[typing.Set, str], socim: sofc.SetOfCims):
        """
        Place in cache the SetOfCims object, and the related sybolyc index parents_comb in _list_of_sets_of_parents

        Parameters:
            parents_comb: the symbolic set index
            socim: the related SetOfCims object

        Returns:
            void
        """
        #print("Putting in _cache:", parents_comb)
        self._list_of_sets_of_parents.append(parents_comb)
        self._actual_cache.append(socim)

    def clear(self):
        """
        Clear the contents of both caches.

        Parameters:
            void
        Returns:
            void
        """
        del self._list_of_sets_of_parents[:]
        del self._actual_cache[:]