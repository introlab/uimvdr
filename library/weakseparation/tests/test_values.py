import unittest

from src.weakseparation import count_values


class TestValues(unittest.TestCase):
    def test_count_values(self):
        values = [-1, 1, 1, 5, 6, -1, 1]
        counts = count_values(values)
        expected_counts = {
            -1: 2,
            1: 3,
            5: 1,
            6: 1
        }
        self.assertEqual(counts, expected_counts)

    def test_count_values_map_abs(self):
        values = [-1, 1, 1, 5, 6, -1, 1]
        counts = count_values(values, abs)
        expected_counts = {
            1: 5,
            5: 1,
            6: 1
        }
        self.assertEqual(counts, expected_counts)


if __name__ == '__main__':
    unittest.main()
