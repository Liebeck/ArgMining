import unittest
import argmining.features.structural_features_spacy as structural_features_spacy
from argmining.models.thf_sentence_export import THFSentenceExport
from argmining.models.token import Token
import argmining.models.thf_sentence_export as models


class THFSentenceFeaturesStructural(unittest.TestCase):
    def test_example1(self):
        tokens = []
        tokens.append(Token(1, b'Kiten', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(2, b'Master', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(3, b':', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(4, b'http://youtu.be/jVVD0OZk-6g', spacy_is_punct=False, spacy_like_url=True))
        sentence_uniqueID = 'p339_s003'
        text = 'Kiten Master: http://youtu.be/jVVD0OZk-6g'
        thf_sentence = THFSentenceExport(sentence_uniqueID, None, text, tokens, None, 1)
        use_sentence_length = True
        feature_value = structural_features_spacy.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [3,
                          0.0 / len(tokens),
                          0.0 / len(tokens),
                          4,
                          1,
                          1,
                          0.0, 0.0, 0.0, 1.0
                          ]
        self.assertEqual(feature_value, expected_value)

    def test_example2(self):
        tokens = []
        tokens.append(Token(1, b'Diesen', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(2, b'Vorschlag', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(3, b'gibt', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(4, b'es', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(5, b'schon', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(6, b':', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(7, b'S\xc3\xbcdlicher', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(8, b'Zugang', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(9, b'zur', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(10, b'Oberlandstra\xc3\x9fe', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(11, b'(', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(12, b'Hatun', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(13, b'-', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(14, b'S\xc3\xbcr\xc3\xbcc\xc3\xbc', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(15, b'-', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(16, b'Br\xc3\xbccke', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(17, b')', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(18, b'(', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(19, b'https://tempelhofer-feld.berlin.de/i/tempelhofer-feld/proposal/104-S%C3%BCdlicher_Zugang_zur_Oberlandstra%C3%9Fe_Hatu', spacy_is_punct=False, spacy_like_url=True))
        tokens.append(Token(20, b')', spacy_is_punct=True, spacy_like_url=False))
        sentence_uniqueID = 'p339_s003'
        text = 'Diesen Vorschlag gibt es schon: S\u00fcdlicher Zugang zur Oberlandstra\u00dfe (Hatun-S\u00fcr\u00fcc\u00fc-Br\u00fccke) (https://tempelhofer-feld.berlin.de/i/tempelhofer-feld/proposal/104-S%C3%BCdlicher_Zugang_zur_Oberlandstra%C3%9Fe_Hatu)'
        thf_sentence = THFSentenceExport(sentence_uniqueID, None, text, tokens, None, 1)
        use_sentence_length = True
        feature_value = structural_features_spacy.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [3,
                          0.0 / len(tokens),
                          0.0 / len(tokens),
                          20,
                          1,
                          7,
                          0.0, 0.0, 0.0, 1.0
                          ]
        self.assertEqual(feature_value, expected_value)

    def test_example3(self):
        tokens = []
        tokens.append(Token(1, b'BM', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(2, b'Tester', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(3, b'#', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(4, b'1', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(5, b':', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(6, b'Kite', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(7, b'-', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(8, b'Skaten', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(9, b'auf', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(10, b'dem', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(11, b'Tempelhofer', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(12, b'Feld', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(13, b':', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(14, b'http://youtu.be/Jf68D61QN4A', spacy_is_punct=False, spacy_like_url=True))
        sentence_uniqueID = 'p339_s003'
        text = 'BM Tester #1: Kite-Skaten auf dem Tempelhofer Feld: http://youtu.be/Jf68D61QN4A'
        thf_sentence = THFSentenceExport(sentence_uniqueID, None, text, tokens, None, 1)
        use_sentence_length = True
        feature_value = structural_features_spacy.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [3,
                          0.0 / len(tokens),
                          0.0 / len(tokens),
                          14,
                          1,
                          4,
                          0.0, 0.0, 0.0, 1.0
                          ]
        self.assertEqual(feature_value, expected_value)

    def test_example4(self):
        tokens = []
        tokens.append(Token(1, b'Hier', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(2, b'eine', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(3, b'Konzept', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(4, b'-', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(5, b'Grafik', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(6, b':', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(7, b' ', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(8, b'http://i.imgur.com/JGlqExO.jpg', spacy_is_punct=False, spacy_like_url=True))
        sentence_uniqueID = 'p339_s003'
        text = 'Hier eine Konzept-Grafik:  http://i.imgur.com/JGlqExO.jpg'
        thf_sentence = THFSentenceExport(sentence_uniqueID, None, text, tokens, None, 1)
        use_sentence_length = True
        feature_value = structural_features_spacy.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [3,
                          0.0 / len(tokens),
                          0.0 / len(tokens),
                          8,
                          1,
                          2,
                          0.0, 0.0, 0.0, 1.0]
        self.assertEqual(feature_value, expected_value)

    def test_example5(self):
        tokens = []
        tokens.append(Token(1, b'Tempelhofparikram', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(2, b'-', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(3, b'Interreligi\xc3\xb6ser', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(4, b'Pilgerpfad', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(5, b'auf', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(6, b'dem', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(7, b'Tempelhofer', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(8, b'Feld', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(9, b'(', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(10, b'http://lebensplan.com/Interreligioeser-Pilgerpfad.pdf', spacy_is_punct=False, spacy_like_url=True))
        tokens.append(Token(11, b')', spacy_is_punct=True, spacy_like_url=False))
        sentence_uniqueID = 'p339_s003'
        text = 'Tempelhofparikram - Interreligi\u00f6ser Pilgerpfad auf dem Tempelhofer Feld (http://lebensplan.com/Interreligioeser-Pilgerpfad.pdf)'
        thf_sentence = THFSentenceExport(sentence_uniqueID, None, text, tokens, None, 1)
        use_sentence_length = True
        feature_value = structural_features_spacy.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [3,
                          0.0 / len(tokens),
                          0.0 / len(tokens),
                          11,
                          1,
                          3,
                          0.0, 0.0, 0.0, 1.0]
        self.assertEqual(feature_value, expected_value)

    def test_example6(self):
        tokens = []
        tokens.append(Token(1, b'kleine', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(2, b'Elektrodrohnen', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(3, b'just', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(4, b'for', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(5, b'fun', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(6, b',', spacy_is_punct=True, spacy_like_url=False))
        tokens.append(Token(7, b'warum', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(8, b'nicht', spacy_is_punct=False, spacy_like_url=False))
        tokens.append(Token(9, b'.', spacy_is_punct=True, spacy_like_url=False))
        sentence_uniqueID = 'p339_s003'
        text = 'kleine Elektrodrohnen just for fun, warum nicht.'
        thf_sentence = THFSentenceExport(sentence_uniqueID, None, text, tokens, None, 1)
        use_sentence_length = True
        feature_value = structural_features_spacy.transform_sentence(thf_sentence, use_sentence_length)
        expected_value = [3,
                          1.0 / len(tokens),
                          1.0 / len(tokens),
                          9,
                          0,
                          2,
                          1.0, 0.0, 0.0, 0.0]
        self.assertEqual(feature_value, expected_value)


if __name__ == '__main__':
    unittest.main()
