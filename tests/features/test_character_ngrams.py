import unittest
from argmining.features.character_ngrams import get_ngrams


class THFSentenceSentiWSAveragePolarity(unittest.TestCase):
    def test_get_ngrams_example1(self):
        tokens = ['Dies', 'ist']
        ngrams_result = get_ngrams(tokens=tokens, min_n=1, max_n=1)
        ngrams_expected = ['D', 'i', 'e', 's', 'i', 's', 't']
        self.assertEqual(ngrams_result, ngrams_expected)

    def test_get_ngrams_example2(self):
        tokens = ['Dies', 'ist']
        ngrams_result = get_ngrams(tokens=tokens, min_n=2, max_n=2)
        ngrams_expected = ['Di', 'ie', 'es', 'is', 'st']
        self.assertEqual(ngrams_result, ngrams_expected)

    def test_get_ngrams_example3(self):
        tokens = ['Regenbogen', 'Feuerwehr']
        ngrams_result = get_ngrams(tokens=tokens, min_n=3, max_n=3)
        ngrams_expected = ['Reg', 'ege', 'gen', 'enb', 'nbo', 'bog', 'oge', 'gen', 'Feu', 'eue', 'uer', 'erw', 'rwe',
                           'weh', 'ehr']
        self.assertEqual(ngrams_result, ngrams_expected)

    def test_get_ngrams_example4(self):
        tokens = ['Regenbogen', 'Feuerwehr']
        ngrams_result = get_ngrams(tokens=tokens, min_n=4, max_n=4)
        ngrams_expected = ['Rege', 'egen', 'genb', 'enbo', 'nbog', 'boge', 'ogen', 'Feue', 'euer', 'uerw', 'erwe',
                           'rweh', 'wehr']
        self.assertEqual(ngrams_result, ngrams_expected)

    def test_get_ngrams_example12(self):
        tokens = ['Dies', 'ist']
        ngrams_result = get_ngrams(tokens=tokens, min_n=1, max_n=2)
        ngrams_expected = ['D', 'i', 'e', 's', 'i', 's', 't', 'Di', 'ie', 'es', 'is', 'st']
        self.assertCountEqual(ngrams_result, ngrams_expected)

    def test_get_ngrams_example34(self):
        tokens = ['Regenbogen', 'Feuerwehr']
        ngrams_result = get_ngrams(tokens=tokens, min_n=3, max_n=4)
        ngrams_expected = ['Rege', 'egen', 'genb', 'enbo', 'nbog', 'boge', 'ogen', 'Feue', 'euer', 'uerw', 'erwe',
                           'rweh', 'wehr', 'Reg', 'ege', 'gen', 'enb', 'nbo', 'bog', 'oge', 'gen', 'Feu', 'eue', 'uer',
                           'erw', 'rwe',
                           'weh', 'ehr']
        self.assertCountEqual(ngrams_result, ngrams_expected)

    if __name__ == '__main__':
        unittest.main()
