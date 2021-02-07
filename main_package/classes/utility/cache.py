import sys
sys.path.append('../')

import typing

import structure_graph.set_of_cims as sofc


class Cache:
    """This class acts as a cache of ``SetOfCims`` objects for a node.

    :_list_of_sets_of_parents: a list of ``Sets`` objects of the parents to which the cim in cache at SAME
        index is related
    :_actual_cache: a list of setOfCims objects
    """

    def __init__(self):
        """Constructor Method
        """
        self.list_of_sets_of_parents = []
        self.actual_cache = []

    def find(self, parents_comb: typing.Set): #typing.Union[typing.Set, str]
        """
        Tries to find in cache given the symbolic parents combination ``parents_comb`` the ``SetOfCims``
        related to that ``parents_comb``.

        :param parents_comb: the parents related to that ``SetOfCims``
        :type parents_comb: Set
        :return: A ``SetOfCims`` object if the ``parents_comb`` index is found in ``_list_of_sets_of_parents``.
            None otherwise.
        :rtype: SetOfCims
        """
        try:
            #print("Cache State:", self.list_of_sets_of_indxs)
            #print("Look For:", parents_comb)
            result = self.actual_cache[self.list_of_sets_of_parents.index(parents_comb)]
            #print("CACHE HIT!!!!", parents_comb)
            return result
        except ValueError:
            return None

    def put(self, parents_comb: typing.Union[typing.Set, str], socim: sofc.SetOfCims):
        """Place in cache the ``SetOfCims`` object, and the related symbolic index ``parents_comb`` in
        ``_list_of_sets_of_parents``.

        :param parents_comb: the symbolic set index
        :type parents_comb: Set
        :param socim: the related SetOfCims object
        :type socim: SetOfCims
        """
        #print("Putting in cache:", parents_comb)
        self.list_of_sets_of_parents.append(parents_comb)
        self.actual_cache.append(socim)

    def clear(self):
        """Clear the contents both of ``_actual_cache`` and ``_list_of_sets_of_parents``.
        """
        del self.list_of_sets_of_parents[:]
        del self.actual_cache[:]