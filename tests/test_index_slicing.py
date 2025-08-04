import unittest
import collections

from sds.index import compute_slicing_bounds

class TestSlicingLogic(unittest.TestCase):
    def test_original_example(self):
        """Tests the primary example from the docstring."""
        counts = {'index1': 10, 'index2': 10, 'index3': 4}
        nodes = 2
        expected = [
            {'index1': (0, 10), 'index2': (0, 2), 'index3': (0, 0)},
            {'index1': (0, 0), 'index2': (2, 10), 'index3': (0, 4)}
        ]
        # The original function passes this test.
        self.assertEqual(compute_slicing_bounds(counts, nodes), expected)

    def test_remainder_handling(self):
        """Tests a case where total samples do not divide evenly."""
        counts = {'a': 5, 'b': 6, 'c': 7} # Total 18
        nodes = 4 # Each gets 4, first two get an extra 1 (5, 5, 4, 4)
        expected = [
            {'a': (0, 5), 'b': (0, 0), 'c': (0, 0)},
            {'a': (0, 0), 'b': (0, 5), 'c': (0, 0)},
            {'a': (0, 0), 'b': (5, 6), 'c': (0, 3)},
            {'a': (0, 0), 'b': (0, 0), 'c': (3, 7)}
        ]
        # The original function passes this test.
        self.assertEqual(compute_slicing_bounds(counts, nodes), expected)

    def test_more_nodes_than_samples(self):
        """Tests where some nodes should get zero samples."""
        counts = {'a': 2, 'b': 1} # Total 3
        nodes = 5 # Should be (1, 1, 1, 0, 0)
        expected = [
            {'a': (0, 1), 'b': (0, 0)},
            {'a': (1, 2), 'b': (0, 0)},
            {'a': (0, 0), 'b': (0, 1)},
            {'a': (0, 0), 'b': (0, 0)},
            {'a': (0, 0), 'b': (0, 0)},
        ]
        # The original function passes this test.
        self.assertEqual(compute_slicing_bounds(counts, nodes), expected)

    def test_zero_samples(self):
        """Tests the edge case of no samples."""
        counts = {'a': 0, 'b': 0}
        nodes = 4
        expected = [
            {'a': (0, 0), 'b': (0, 0)},
            {'a': (0, 0), 'b': (0, 0)},
            {'a': (0, 0), 'b': (0, 0)},
            {'a': (0, 0), 'b': (0, 0)},
        ]
        self.assertEqual(compute_slicing_bounds(counts, nodes), expected)

    def test_complex_distribution(self):
        """A more difficult test case that splits a large data source."""
        # Using OrderedDict to ensure key order for the proposed function
        counts = collections.OrderedDict([
            ('ds1', 17), ('ds2', 3), ('ds3', 91), ('ds4', 6), ('ds5', 1)
        ])
        nodes = 2
        # Total samples = 118. Each node gets 59.
        # Node 0 gets all of ds1 (17), ds2 (3), and the first 39 of ds3. (17+3+39=59)
        # Node 1 gets the rest of ds3 (52), all of ds4 (6) and ds5 (1). (52+6+1=59)
        expected = [
            {'ds1': (0, 17), 'ds2': (0, 3), 'ds3': (0, 39), 'ds4': (0, 0), 'ds5': (0, 0)},
            {'ds1': (0, 0), 'ds2': (0, 0), 'ds3': (39, 91), 'ds4': (0, 6), 'ds5': (0, 1)}
        ]
        self.assertEqual(compute_slicing_bounds(counts, nodes), expected)
