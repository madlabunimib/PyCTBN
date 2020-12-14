
import unittest
import numpy as np

from ..classes.cache import Cache
from ..classes.set_of_cims import SetOfCims


class TestCache(unittest.TestCase):

    def test_init(self):
        c1 = Cache()
        self.assertFalse(c1._list_of_sets_of_parents)
        self.assertFalse(c1._actual_cache)

    def test_put(self):
        c1 = Cache()
        pset1 = {'X', 'Y'}
        sofc1 = SetOfCims('Z', [], 3, np.array([]))
        c1.put(pset1, sofc1)
        self.assertEqual(1, len(c1._actual_cache))
        self.assertEqual(1, len(c1._list_of_sets_of_parents))
        self.assertEqual(sofc1, c1._actual_cache[0])
        pset2 = {'X'}
        sofc2 = SetOfCims('Z', [], 3, np.array([]))
        c1.put(pset2, sofc2)
        self.assertEqual(2, len(c1._actual_cache))
        self.assertEqual(2, len(c1._list_of_sets_of_parents))
        self.assertEqual(sofc2, c1._actual_cache[1])

    def test_find(self):
        c1 = Cache()
        pset1 = {'X', 'Y'}
        sofc1 = SetOfCims('Z', [], 3, np.array([]))
        c1.put(pset1, sofc1)
        self.assertEqual(1, len(c1._actual_cache))
        self.assertEqual(1, len(c1._list_of_sets_of_parents))
        self.assertIsInstance(c1.find(pset1), SetOfCims)
        self.assertEqual(sofc1, c1.find(pset1))
        self.assertIsInstance(c1.find({'Y', 'X'}), SetOfCims)
        self.assertEqual(sofc1, c1.find({'Y', 'X'}))
        self.assertIsNone(c1.find({'X'}))

    def test_clear(self):
        c1 = Cache()
        pset1 = {'X', 'Y'}
        sofc1 = SetOfCims('Z', [], 3, np.array([]))
        c1.put(pset1, sofc1)
        self.assertEqual(1, len(c1._actual_cache))
        self.assertEqual(1, len(c1._list_of_sets_of_parents))
        c1.clear()
        self.assertFalse(c1._list_of_sets_of_parents)
        self.assertFalse(c1._actual_cache)


if __name__ == '__main__':
    unittest.main()
