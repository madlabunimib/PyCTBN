import typing
import set_of_cims as sofc


class Cache:
    """
    This class has the role of a cache for SetOfCIMS of a test node that have been already computed during the ctpc algorithm.

    :list_of_sets_of_parents: a list of Sets of the parents to which the cim in cache at SAME index is related
    :actual_cache: a list of setOfCims objects
    """

    def __init__(self):
        self.list_of_sets_of_parents = []
        self.actual_cache = []

    def find(self, parents_comb: typing.Union[typing.Set, str]):
        """
        Tries to find in cache given the symbolic parents combination parents_comb the SetOfCims related to that parents_comb.
        N.B. if the parents_comb is not a Set, than the index refers to the SetOfCims of the actual node with no parents.
        Parameters:
            parents_comb: the parents related to that SetOfCims
        Returns:
            A SetOfCims object if the parents_comb index is found in list_of_sets_of_parents.
            None otherwise.

        """
        try:
            #print("Cache State:", self.list_of_sets_of_indxs)
            #print("Look For:", parents_comb)
            result = self.actual_cache[self.list_of_sets_of_parents.index(parents_comb)]
            print("CACHE HIT!!!!", parents_comb)
            return result
        except ValueError:
            return None

    def put(self, parents_comb: typing.Union[typing.Set, str], socim: sofc.SetOfCims):
        """
        Place in cache the SetOfCims object, and the related sybolyc index parents_comb in list_of_sets_of_parents

        Parameters:
            parents_comb: the symbolic set index
            socim: the related SetOfCims object

        Returns:
            void
        """
        #print("Putting in cache:", parents_comb)
        self.list_of_sets_of_parents.append(parents_comb)
        self.actual_cache.append(socim)

    def clear(self):
        """
        Clear the contents of both caches.

        Parameters:
            void
        Returns:
            void
        """
        del self.list_of_sets_of_parents[:]
        del self.actual_cache[:]