import gensim
from gensim.models import word2vec
import logging


class Word2Vec:
    def __init__(self, model_path='data/word_embeddings/word2vec_wiki-de_20161120_100', text_type='lowercase'):
        self.model_path = model_path
        self.text_type = text_type
        self.logger = logging.getLogger()
        self.coverage = 0
        self.total_tokens = 0

    def load(self):
        self.logger.info("Loading model: {}".format(self.model_path))
        self.logger.info("text_type: {}".format(self.text_type))
        self.model = gensim.models.Word2Vec.load_word2vec_format(self.model_path)
        self.logger.info("Model loaded")

    def get_key(self, token):
        if self.text_type == 'lowercase':
            return token.text.lower()
        return token.text

    def annotate_sentence(self, sentence):
        for token in sentence.tokens:
            self.total_tokens = self.total_tokens + 1
            key = self.get_key(token)
            if key in self.model.vocab:
                self.coverage = self.coverage + 1
                # self.logger.debug(type(self.model[key]))
                token.embedding = self.model[key]

    def annotate_sentences(self, sentences):
        list(map(lambda sentence: self.annotate_sentence(sentence), sentences))
        self.logger.info("Annotated Tokens {}/{}".format(self.coverage, self.total_tokens))
