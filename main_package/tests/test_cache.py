import unittest
import numpy as np

import cache as ch
import set_of_cims as soci


class TestCache(unittest.TestCase):

    def test_init(self):
        c1 = ch.Cache()
        self.assertFalse(c1.list_of_sets_of_parents)
        self.assertFalse(c1.actual_cache)

    def test_put(self):
        c1 = ch.Cache()
        pset1 = {'X', 'Y'}
        sofc1 = soci.SetOfCims('Z', [], 3, np.array([]))
        c1.put(pset1, sofc1)
        self.assertEqual(1, len(c1.actual_cache))
        self.assertEqual(1, len(c1.list_of_sets_of_parents))
        self.assertEqual(sofc1, c1.actual_cache[0])
        pset2 = {'X'}
        sofc2 = soci.SetOfCims('Z', [], 3, np.array([]))
        c1.put(pset2, sofc2)
        self.assertEqual(2, len(c1.actual_cache))
        self.assertEqual(2, len(c1.list_of_sets_of_parents))
        self.assertEqual(sofc2, c1.actual_cache[1])

    def test_find(self):
        c1 = ch.Cache()
        pset1 = {'X', 'Y'}
        sofc1 = soci.SetOfCims('Z', [], 3, np.array([]))
        c1.put(pset1, sofc1)
        self.assertEqual(1, len(c1.actual_cache))
        self.assertEqual(1, len(c1.list_of_sets_of_parents))
        self.assertIsInstance(c1.find(pset1), soci.SetOfCims)
        self.assertEqual(sofc1, c1.find(pset1))
        self.assertIsInstance(c1.find({'Y', 'X'}), soci.SetOfCims)
        self.assertEqual(sofc1, c1.find({'Y', 'X'}))
        self.assertIsNone(c1.find({'X'}))

    def test_clear(self):
        c1 = ch.Cache()
        pset1 = {'X', 'Y'}
        sofc1 = soci.SetOfCims('Z', [], 3, np.array([]))
        c1.put(pset1, sofc1)
        self.assertEqual(1, len(c1.actual_cache))
        self.assertEqual(1, len(c1.list_of_sets_of_parents))
        c1.clear()
        self.assertFalse(c1.list_of_sets_of_parents)
        self.assertFalse(c1.actual_cache)







if __name__ == '__main__':
    unittest.main()
