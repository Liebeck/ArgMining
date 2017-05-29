import unittest
from scripts.nlp.sentiws_wrapper import SentiWSWrapper


class SentiWSWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.sentiws = SentiWSWrapper(sentiws_path='data/sentiws')

    def test_contains_example1(self):
        self.assertEqual(self.sentiws.contains('Depression', 'NOUN'), True)

    def test_contains_example2(self):
        self.assertEqual(self.sentiws.contains('Depressionen', 'NOUN'), True)

    def test_contains_example3(self):
        self.assertEqual(self.sentiws.contains('Demütigung', 'NOUN'), True)

    def test_contains_example4(self):
        self.assertEqual(self.sentiws.contains('ablaufen', 'VERB'), True)

    def test_contains_example5(self):
        self.assertEqual(self.sentiws.contains('abfällig', 'ADJ'), True)

    def test_contains_example6(self):
        self.assertEqual(self.sentiws.contains('abfällig', 'ADV'), False)

    def test_contains_example7(self):
        self.assertEqual(self.sentiws.contains('bergab', 'ADV'), True)

    def test_contains_example8(self):
        self.assertEqual(self.sentiws.contains('Anheiterung', 'NOUN'), True)

    def test_determine_example1(self):
        self.assertEqual(self.sentiws.determine('Anheiterung', 'NOUN'), 0.004)

    def test_determine_example2(self):
        self.assertEqual(self.sentiws.determine('anheiterung', 'NOUN'), 0.004)

    def test_determine_example3(self):
        self.assertEqual(self.sentiws.determine('.', 'PUNCT'), None)

    def test_determine_example4(self):
        self.assertEqual(self.sentiws.determine('BEKÜMMERT', 'VERB'), -0.4503)


if __name__ == '__main__':
    unittest.main()
