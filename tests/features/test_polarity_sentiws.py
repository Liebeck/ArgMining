import unittest
import argmining.features.polarity_sentiws_feature as polarity_sentiws_feature
from argmining.models.thf_sentence_export import THFSentenceExport
from argmining.models.token import Token


class THFSentenceFeaturesStructural(unittest.TestCase):
    def test_extract_average_polarity_example1(self):
        tokens = []
        tokens.append(Token(1, None, None, None, None, None, None, 0.5))
        tokens.append(Token(2, None, None, None, None, None, None, None))
        tokens.append(Token(3, None, None, None, None, None, None, 1.5))
        thf_sentence = THFSentenceExport(None, None, None, tokens, None)
        feature_value = polarity_sentiws_feature.extract_average_polarity(thf_sentence)
        expected_value = [1.0]
        self.assertEqual(feature_value, expected_value)

    def test_extract_average_polarity_example2(self):
        tokens = []
        tokens.append(Token(1, None, None, None, None, None, None, None))
        tokens.append(Token(2, None, None, None, None, None, None, None))
        tokens.append(Token(3, None, None, None, None, None, None, None))
        thf_sentence = THFSentenceExport(None, None, None, tokens, None)
        feature_value = polarity_sentiws_feature.extract_average_polarity(thf_sentence)
        expected_value = [0.0]
        self.assertEqual(feature_value, expected_value)

    def test_extract_average_polarity_example3(self):
        tokens = []
        tokens.append(Token(1, None, None, None, None, None, None, None))
        tokens.append(Token(2, None, None, None, None, None, None, -1))
        tokens.append(Token(3, None, None, None, None, None, None, -1.5))
        thf_sentence = THFSentenceExport(None, None, None, tokens, None)
        feature_value = polarity_sentiws_feature.extract_average_polarity(thf_sentence)
        expected_value = [-1.25]
        self.assertEqual(feature_value, expected_value)



if __name__ == '__main__':
    unittest.main()
