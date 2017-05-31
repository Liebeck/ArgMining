import unittest
import argmining.features.ner_feature as ner_feature
from argmining.models.thf_sentence_export import THFSentenceExport
from argmining.models.token import Token
import numpy as np


class THFSentenceSentiWSAveragePolarity(unittest.TestCase):
    def test_count_different_ner_labels_example1(self):
        tokens = []
        tokens.append(Token(1, b'Wenn', spacy_pos_universal_google='SCONJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(2, b'ich', spacy_pos_universal_google='PRON', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(3, b'durch', spacy_pos_universal_google='ADP', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(4, b'den', spacy_pos_universal_google='DET', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(5, b'Hans', spacy_pos_universal_google='PROPN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(6, b'-', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(7, b'Baluschek', spacy_pos_universal_google='PROPN', spacy_ner_type='PERSON', spacy_ner_iob='B'))
        tokens.append(Token(8, b'-', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(9, b'Park', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(10, b'radle', spacy_pos_universal_google='VERB', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(11, b',', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(12, b'riecht', spacy_pos_universal_google='VERB', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(13, b'es', spacy_pos_universal_google='PRON', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(14, b'immer', spacy_pos_universal_google='ADV', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(15, b'stark', spacy_pos_universal_google='ADJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(16, b'vom', spacy_pos_universal_google='ADP', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(17, b'angrenzenden', spacy_pos_universal_google='ADJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(18, b'S\xc3\xbcdgel\xc3\xa4nde', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(19, b'nach', spacy_pos_universal_google='ADP', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(20, b'Farbd\xc3\xbcnsten', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(21, b'und', spacy_pos_universal_google='CONJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(22, b'das', spacy_pos_universal_google='PRON', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(23, b'passt', spacy_pos_universal_google='VERB', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(24, b'nicht', spacy_pos_universal_google='PART', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(25, b'wirklich', spacy_pos_universal_google='ADJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(26, b'zum', spacy_pos_universal_google='ADP', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(27, b'Naturschutz', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(28, b'.', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))

        thf_sentence = THFSentenceExport(None, None,
                                         "Wenn ich durch den Hans-Baluschek-Park radle, riecht es immer stark vom angrenzenden Südgelände nach Farbdünsten und das passt nicht wirklich zum Naturschutz.",
                                         tokens, None, 1)
        feature_value = ner_feature.count_different_ner_labels(thf_sentence.tokens)
        expected_value = np.array([1, 0, 0], dtype=np.float64)
        self.assertEqual(np.array_equal(feature_value, expected_value), True)

    def test_count_different_ner_labels_example2(self):
        tokens = []
        tokens.append(Token(1, b'Bei', spacy_pos_universal_google='ADP', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(2, b'der', spacy_pos_universal_google='DET', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(3, b'Stauraumplanung', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(4, b'wird', spacy_pos_universal_google='AUX', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(5, b'es', spacy_pos_universal_google='PRON', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(6, b'aus', spacy_pos_universal_google='ADP', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(7, b'allen', spacy_pos_universal_google='DET', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(8, b'Gullis', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(9, b'demn\xc3\xa4chst', spacy_pos_universal_google='ADV', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(10, b'stinken', spacy_pos_universal_google='VERB', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(11, b'und', spacy_pos_universal_google='CONJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(12, b'Abwaaserlagerung', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(13, b'in', spacy_pos_universal_google='ADP', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(14, b'Schiffen', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(15, b'auf', spacy_pos_universal_google='ADP', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(16, b'der', spacy_pos_universal_google='DET', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(17, b'Spree', spacy_pos_universal_google='PROPN', spacy_ner_type='LOC', spacy_ner_iob='B'))
        tokens.append(Token(18, b'd\xc3\xbcrfte', spacy_pos_universal_google='VERB', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(19, b'auch', spacy_pos_universal_google='ADV', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(20, b'nicht', spacy_pos_universal_google='PART', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(21, b'gesund', spacy_pos_universal_google='ADJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(22, b'sein', spacy_pos_universal_google='AUX', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(23, b'.', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))
        thf_sentence = THFSentenceExport(None, None,
                                         "Bei der Stauraumplanung wird es aus allen Gullis demnächst stinken und Abwaaserlagerung in Schiffen auf der Spree dürfte auch nicht gesund sein.",
                                         tokens, None, 1)
        feature_value = ner_feature.count_different_ner_labels(thf_sentence.tokens)
        expected_value = np.array([0, 1, 0], dtype=np.float64)
        self.assertEqual(np.array_equal(feature_value, expected_value), True)

    def test_count_different_ner_labels_example3(self):
        tokens = []
        tokens.append(Token(1, b'Auf', spacy_pos_universal_google='ADP', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(2, b'dem', spacy_pos_universal_google='DET', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(3, b'Tempelhofer', spacy_pos_universal_google='ADJ', spacy_ner_type='LOC', spacy_ner_iob='B'))
        tokens.append(Token(4, b'Feld', spacy_pos_universal_google='NOUN', spacy_ner_type='LOC', spacy_ner_iob='I'))
        tokens.append(Token(5, b'stehen', spacy_pos_universal_google='VERB', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(6, b'22', spacy_pos_universal_google='NUM', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(7, b'kleinere', spacy_pos_universal_google='ADJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(8, b'Geb\xc3\xa4ude', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(9, b',', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(10, b'gr\xc3\xb6\xc3\x9ftenteils', spacy_pos_universal_google='ADV', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(11, b'ungenutzt', spacy_pos_universal_google='ADJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(12, b'.', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))
        thf_sentence = THFSentenceExport(None, None,
                                         "Auf dem Tempelhofer Feld stehen 22 kleinere Gebäude, größtenteils ungenutzt.",
                                         tokens, None, 1)
        feature_value = ner_feature.count_different_ner_labels(thf_sentence.tokens)
        expected_value = np.array([0, 1, 0], dtype=np.float64)
        self.assertEqual(np.array_equal(feature_value, expected_value), True)

    def test_count_different_ner_labels_example4(self):
        tokens = []
        tokens.append(Token(1, b'Oder', spacy_pos_universal_google='CONJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(2, b',', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(3, b'um', spacy_pos_universal_google='SCONJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(4, b'mit', spacy_pos_universal_google='ADP', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(5, b'Hermann', spacy_pos_universal_google='PROPN', spacy_ner_type='PERSON', spacy_ner_iob='B'))
        tokens.append(Token(6, b'Hesse', spacy_pos_universal_google='PROPN', spacy_ner_type='PERSON', spacy_ner_iob='I'))
        tokens.append(Token(7, b'zu', spacy_pos_universal_google='PART', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(8, b'sprechen', spacy_pos_universal_google='VERB', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(9, b':', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(10, b'Jedem', spacy_pos_universal_google='DET', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(11, b'Ende', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(12, b'wohnt', spacy_pos_universal_google='VERB', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(13, b'ein', spacy_pos_universal_google='DET', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(14, b'neuer', spacy_pos_universal_google='ADJ', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(15, b'Anfang', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(16, b'inne', spacy_pos_universal_google='PART', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(17, b'.', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))
        thf_sentence = THFSentenceExport(None, None,
                                         "Oder, um mit Hermann Hesse zu sprechen: Jedem Ende wohnt ein neuer Anfang inne.",
                                         tokens, None, 1)
        feature_value = ner_feature.count_different_ner_labels(thf_sentence.tokens)
        expected_value = np.array([1, 0, 0], dtype=np.float64)
        self.assertEqual(np.array_equal(feature_value, expected_value), True)

    def test_count_different_ner_labels_example5(self):
        tokens = []
        tokens.append(Token(1, b'Vorbild', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(2, b'New', spacy_pos_universal_google='PROPN', spacy_ner_type='LOC', spacy_ner_iob='B'))
        tokens.append(Token(3, b'York', spacy_pos_universal_google='PROPN', spacy_ner_type='LOC', spacy_ner_iob='I'))
        tokens.append(Token(4, b':', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(5, b'http://www.houndsandpeople.com/de/magazin/kultur/new-york-city-dogs-teil-und-seele-der-weltmetropole/', spacy_pos_universal_google='AUX', spacy_ner_type='', spacy_ner_iob='O'))
        thf_sentence = THFSentenceExport(None, None,
                                         "Vorbild New York: http://www.houndsandpeople.com/de/magazin/kultur/new-york-city-dogs-teil-und-seele-der-weltmetropole/",
                                         tokens, None, 1)
        feature_value = ner_feature.count_different_ner_labels(thf_sentence.tokens)
        expected_value = np.array([0, 1, 0], dtype=np.float64)
        self.assertEqual(np.array_equal(feature_value, expected_value), True)


if __name__ == '__main__':
    unittest.main()
