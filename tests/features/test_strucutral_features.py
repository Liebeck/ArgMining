import unittest
import argmining.features.structural_features as structural_features
from argmining.models.thf_sentence_export import THFSentenceExport
from argmining.models.token import Token
import argmining.models.thf_sentence_export as models


class THFSentenceFeaturesStructural(unittest.TestCase):
    def test_example1(self):
        tokens = []
        tokens.append(Token(1, 'Das', None, None, None, None, None, None))
        tokens.append(Token(2, 'ist', None, None, None, None, None, None))
        tokens.append(Token(3, 'ein', None, None, None, None, None, None))
        tokens.append(Token(4, ',', None, None, None, None, None, None))
        tokens.append(Token(5, 'Test', None, None, None, None, None, None))
        tokens.append(Token(6, '!', None, None, None, None, None, None))
        thf_sentence = THFSentenceExport(None, None, 'Das ist ein , Test!', tokens, None)
        use_sentence_length = True
        feature_value = structural_features.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [1.0 / len(tokens),
                          0.0 / len(tokens),
                          1.0 * len(tokens),
                          0.0,
                          0.0, 1.0, 0.0, 0.0
                          ]
        self.assertEqual(feature_value, expected_value)

    def test_example2(self):
        tokens = []
        tokens.append(Token(1, 'Das', None, None, None, None, None, None))
        tokens.append(Token(2, 'ist', None, None, None, None, None, None))
        tokens.append(Token(3, '.', None, None, None, None, None, None))
        tokens.append(Token(4, ',', None, None, None, None, None, None))
        tokens.append(Token(5, 'Test', None, None, None, None, None, None))
        tokens.append(Token(6, '!', None, None, None, None, None, None))
        thf_sentence = THFSentenceExport(None, None, 'Das ist . , Test!', tokens, None)
        use_sentence_length = True
        feature_value = structural_features.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [1.0 / len(tokens),
                          1.0 / len(tokens),
                          1.0 * len(tokens),
                          0.0,
                          0.0, 1.0, 0.0, 0.0
                          ]
        self.assertEqual(feature_value, expected_value)

    def test_example2_without_sentence_length(self):
        tokens = []
        tokens.append(Token(1, 'Das', None, None, None, None, None, None))
        tokens.append(Token(2, 'ist', None, None, None, None, None, None))
        tokens.append(Token(3, '.', None, None, None, None, None, None))
        tokens.append(Token(4, ',', None, None, None, None, None, None))
        tokens.append(Token(5, 'Test', None, None, None, None, None, None))
        tokens.append(Token(6, '!', None, None, None, None, None, None))
        thf_sentence = THFSentenceExport(None, None, 'Das ist . , Test!', tokens, None)
        use_sentence_length = False
        feature_value = structural_features.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [1.0 / len(tokens),
                          1.0 / len(tokens),
                          0.0,
                          0.0, 1.0, 0.0, 0.0
                          ]
        self.assertEqual(feature_value, expected_value)

    def test_comma_end(self):
        tokens = []
        tokens.append(Token(1, 'Test', None, None, None, None, None, None))
        tokens.append(Token(2, ',', None, None, None, None, None, None))
        thf_sentence = THFSentenceExport(None, None, 'Test ,', tokens, None)
        use_sentence_length = False
        feature_value = structural_features.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [1.0 / len(tokens),
                          0.0 / len(tokens),
                          0.0,
                          0.0, 0.0, 0.0, 1.0
                          ]
        self.assertEqual(feature_value, expected_value)

    def test_dot_end(self):
        tokens = []
        tokens.append(Token(1, 'Test', None, None, None, None, None, None))
        tokens.append(Token(2, '.', None, None, None, None, None, None))
        thf_sentence = THFSentenceExport(None, None, 'Test .', tokens, None)
        use_sentence_length = False
        feature_value = structural_features.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [0.0 / len(tokens),
                          1.0 / len(tokens),
                          0.0,
                          1.0, 0.0, 0.0, 0.0
                          ]
        self.assertEqual(feature_value, expected_value)

    def test_exclamation_mark_end(self):
        tokens = []
        tokens.append(Token(1, 'Test', None, None, None, None, None, None))
        tokens.append(Token(2, '!', None, None, None, None, None, None))
        thf_sentence = THFSentenceExport(None, None, 'Test !', tokens, None)
        use_sentence_length = False
        feature_value = structural_features.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [0.0 / len(tokens),
                          0.0 / len(tokens),
                          0.0,
                          0.0, 1.0, 0.0, 0.0
                          ]
        self.assertEqual(feature_value, expected_value)

    def test_question_mark_end(self):
        tokens = []
        tokens.append(Token(1, 'Test', None, None, None, None, None, None))
        tokens.append(Token(2, '?', None, None, None, None, None, None))
        thf_sentence = THFSentenceExport(None, None, 'Test ?', tokens, None)
        use_sentence_length = False
        feature_value = structural_features.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [0.0 / len(tokens),
                          0.0 / len(tokens),
                          0.0,
                          0.0, 0.0, 1.0, 0.0
                          ]
        self.assertEqual(feature_value, expected_value)


if __name__ == '__main__':
    unittest.main()
