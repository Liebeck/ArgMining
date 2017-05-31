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
        tokens.append(
            Token(27, b'Naturschutz', spacy_pos_universal_google='NOUN', spacy_ner_type='', spacy_ner_iob='O'))
        tokens.append(Token(28, b'.', spacy_pos_universal_google='PUNCT', spacy_ner_type='', spacy_ner_iob='O'))

        thf_sentence = THFSentenceExport(None, None,
                                         "Wenn ich durch den Hans-Baluschek-Park radle, riecht es immer stark vom angrenzenden Südgelände nach Farbdünsten und das passt nicht wirklich zum Naturschutz.",
                                         tokens, None, 1)
        feature_value = ner_feature.count_different_ner_labels(thf_sentence.tokens)
        expected_value = np.array([1, 0, 0], dtype=np.float64)
        self.assertEqual(np.array_equal(feature_value, expected_value), True)


if __name__ == '__main__':
    unittest.main()
