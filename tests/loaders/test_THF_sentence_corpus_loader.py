import unittest
from argmining.sentence.loaders.THF_sentence_corpus_loader import parse_tree_tagger_lemma


class THFSentenceCorpusLoaderTests(unittest.TestCase):
    def test_tree_tagger_lemma_empty(self):
        parsed_lemma = parse_tree_tagger_lemma('aa')
        self.assertEqual(parsed_lemma, None)


if __name__ == '__main__':
    unittest.main()


# Todo: 2 unit tests
# <TreeTaggerLemma>
# <string>werden</string>
# </TreeTaggerLemma>
# <TreeTaggerLemma />
