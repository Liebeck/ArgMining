import gensim
import logging
from gensim.models.ldamodel import LdaModel
from gensim import corpora
import bz2
import numpy as np


class LDA:
    def __init__(self, model_path='data/lda/THF/lda_20', vocab_path='data/lda/THF/lda_20_wordids.txt.bz2',
                 nouns_only=True):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.nouns_only = nouns_only
        self.logger = logging.getLogger()

    def load(self):
        self.lda_model = LdaModel.load(self.model_path)
        self.vocab = corpora.Dictionary.load_from_text(bz2.BZ2File(self.vocab_path))

    def infer_topics(self, tokens):
        # print([x.encode('utf-8') for x in tokens])
        vec_bow = self.vocab.doc2bow(tokens)
        topics = self.lda_model.get_document_topics(vec_bow, per_word_topics=False,
                                                    minimum_probability=0)
        probabilities = np.asarray([topic[1] for topic in topics])
        return probabilities

    def annotate_sentence(self, sentence):
        tokens_for_lda = []
        for token in sentence.tokens:
            if self.nouns_only and token.spacy_pos_universal_google != 'NOUN':
                continue
            if token.iwnlp_lemma is not None and len(token.iwnlp_lemma) == 1:
                tokens_for_lda.append(token.iwnlp_lemma[0].lower())
            else:
                tokens_for_lda.append(token.text.lower())
        sentence.lda_embedding = self.infer_topics(tokens_for_lda)

    def annotate_sentences(self, sentences):
        for sentence in sentences:
            self.annotate_sentence(sentence)
        self.logger.info("Annotated all sentences with LDA probabilities")
