import sys
sys.path.append("./classes/")
import typing
import set_of_cims as sofc


class Cache:
    """This class acts as a cache of ``SetOfCims`` objects for a node.

    :_list_of_sets_of_parents: a list of ``Sets`` objects of the parents to which the cim in cache at SAME
        index is related
    :_actual_cache: a list of setOfCims objects
    """

    def __init__(self):
        """Constructor Method
        """
        self._list_of_sets_of_parents = []
        self._actual_cache = []

    def find(self, parents_comb: typing.Set) -> sofc.SetOfCims:
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
            result = self._actual_cache[self._list_of_sets_of_parents.index(parents_comb)]
            return result
        except ValueError:
            return None

    def put(self, parents_comb: typing.Set, socim: sofc.SetOfCims) -> None:
        """Place in cache the ``SetOfCims`` object, and the related symbolic index ``parents_comb`` in
        ``_list_of_sets_of_parents``.

        :param parents_comb: the symbolic set index
        :type parents_comb: Set
        :param socim: the related SetOfCims object
        :type socim: SetOfCims
        """
        self._list_of_sets_of_parents.append(parents_comb)
        self._actual_cache.append(socim)

    def clear(self) -> None:
        """Clear the contents both of ``_actual_cache`` and ``_list_of_sets_of_parents``.
        """
        del self._list_of_sets_of_parents[:]
        del self._actual_cache[:]