import unittest
import argmining.features.sentiws_polarity_bearing_tokens_feature as sentiws_polarity_bearing_tokens_feature
from argmining.models.thf_sentence_export import THFSentenceExport
from argmining.models.token import Token


class THFSentenceSentiWSPolarityBearingTokens(unittest.TestCase):
    def test_count_polarity_bearing_tokens_example1(self):
        tokens = []
        tokens.append(Token(1, None, None, None, None, None, None, 0.5))
        tokens.append(Token(2, None, None, None, None, None, None, None))
        tokens.append(Token(3, None, None, None, None, None, None, 1.5))
        thf_sentence = THFSentenceExport(None, None, None, tokens, None)
        feature_value = sentiws_polarity_bearing_tokens_feature.count_polarity_bearing_tokens(thf_sentence)
        expected_value = [2]
        self.assertEqual(feature_value, expected_value)

    def test_count_polarity_bearing_tokens_example2(self):
        tokens = []
        tokens.append(Token(1, None, None, None, None, None, None, None))
        tokens.append(Token(2, None, None, None, None, None, None, None))
        tokens.append(Token(3, None, None, None, None, None, None, None))
        thf_sentence = THFSentenceExport(None, None, None, tokens, None)
        feature_value = sentiws_polarity_bearing_tokens_feature.count_polarity_bearing_tokens(thf_sentence)
        expected_value = [0]
        self.assertEqual(feature_value, expected_value)

    def test_count_polarity_bearing_tokens_example3(self):
        tokens = []
        tokens.append(Token(1, None, None, None, None, None, None, None))
        tokens.append(Token(2, None, None, None, None, None, None, -1))
        tokens.append(Token(3, None, None, None, None, None, None, -1.5))
        thf_sentence = THFSentenceExport(None, None, None, tokens, None)
        feature_value = sentiws_polarity_bearing_tokens_feature.count_polarity_bearing_tokens(thf_sentence)
        expected_value = [2]
        self.assertEqual(feature_value, expected_value)


if __name__ == '__main__':
    unittest.main()
