import unittest
from argmining.sentence.loaders.THF_sentence_corpus_loader import parse_tree_tagger_lemma
from argmining.sentence.loaders.THF_sentence_corpus_loader import parse_IWNLP_lemma
from argmining.sentence.loaders.THF_sentence_corpus_loader import parse_polarity


class THFSentenceCorpusLoaderTests(unittest.TestCase):
    def test_tree_tagger_lemma_empty(self):
        input_value = []
        parsed_lemma = parse_tree_tagger_lemma(input_value)
        self.assertEqual(parsed_lemma, None)

    def test_tree_tagger_lemma_one_value(self):
        input_value = ["flächendeckend"]
        parsed_lemma = parse_tree_tagger_lemma(input_value)
        self.assertEqual(parsed_lemma, ["flächendeckend"])

    def test_tree_tagger_lemma_two_values(self):
        input_value = ["Rüge", "Rügen"]
        parsed_lemma = parse_IWNLP_lemma(input_value)
        self.assertEqual(parsed_lemma, ["Rüge", "Rügen"])

    def test_iwnlp_lemma_empty(self):
        input_value = []
        parsed_lemma = parse_IWNLP_lemma(input_value)
        self.assertEqual(parsed_lemma, None)

    def test_iwnlp_lemma_one_value(self):
        input_value = ["Ferkel"]
        parsed_lemma = parse_IWNLP_lemma(input_value)
        self.assertEqual(parsed_lemma, ["Ferkel"])

    def test_iwnlp_lemma_two_values(self):
        input_value = ["Rügen", "Rüge"]
        parsed_lemma = parse_IWNLP_lemma(input_value)
        self.assertEqual(parsed_lemma, ["Rügen", "Rüge"])

    def test_polarity_empty(self):
        input_value = None
        parsed_polarity = parse_polarity(input_value)
        self.assertEqual(parsed_polarity, None)

    def test_polarity_with_value(self):
        input_value = 0.004
        parsed_polarity = parse_polarity(input_value)
        self.assertEqual(parsed_polarity, 0.004)


if __name__ == '__main__':
    unittest.main()
