import unittest

from src.weakseparation import count_characters, count_characters_ignoreCase


class TestCharacters(unittest.TestCase):
    def test_count_characters(self):
        counts = count_characters('Bonjour Bob')
        expected_counts = {
            'B': 2,
            'o': 3,
            'n': 1,
            'j': 1,
            'u': 1,
            'r': 1,
            ' ': 1,
            'b': 1
        }
        self.assertEqual(counts, expected_counts)

    def test_count_characters_ignore_case(self):
        counts = count_characters_ignoreCase('BonjOur BOb')
        expected_counts = {
            'B': 3,
            'O': 3,
            'N': 1,
            'J': 1,
            'U': 1,
            'R': 1,
            ' ': 1
        }
        self.assertEqual(counts, expected_counts)


if __name__ == '__main__':
    unittest.main()
