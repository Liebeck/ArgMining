import unittest
from argmining.sentence.loaders.THF_sentence_corpus_loader import parse_tree_tagger_lemma
from argmining.sentence.loaders.THF_sentence_corpus_loader import parse_IWNLP_lemma


class THFSentenceCorpusLoaderTests(unittest.TestCase):
    def test_tree_tagger_lemma_empty(self):
        input_value = []
        parsed_lemma = parse_tree_tagger_lemma(input_value)
        self.assertEqual(parsed_lemma, None)

    def test_tree_tagger_lemma_one_value(self):
        input_value = ["flächendeckend"]
        parsed_lemma = parse_tree_tagger_lemma(input_value)
        self.assertEqual(parsed_lemma, "flächendeckend")

    def test_iwnlp_lemma_empty(self):
        input_value = []
        parsed_lemma = parse_IWNLP_lemma(input_value)
        self.assertEqual(parsed_lemma, None)

    def test_iwnlp_lemma_one_value(self):
        input_value = ["Ferkel"]
        parsed_lemma = parse_IWNLP_lemma(input_value)
        self.assertEqual(parsed_lemma, "Ferkel")


if __name__ == '__main__':
    unittest.main()
