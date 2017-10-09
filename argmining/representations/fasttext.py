import gensim
import logging
from gensim.models.wrappers.fasttext import FastText as FT_wrapper


class FastText:
    def __init__(self, model_path='data/fasttext/dewiki-20170501-3_3-5'):
        self.model_path = model_path
        self.logger = logging.getLogger()
        self.coverage = 0
        self.total_tokens = 0
        self.text_type = 'text'

    def load(self):
        self.logger.info("Loading model: {}".format(self.model_path))
        self.model = FT_wrapper.load(self.model_path)
        self.logger.info("Model loaded")

    def annotate_sentence(self, sentence):
        for token in sentence.tokens:
            self.total_tokens = self.total_tokens + 1
            key = token.get_key(self.text_type)
            if key in self.model:
                self.coverage = self.coverage + 1
                token.character_embedding = self.model[key]

    def annotate_sentences(self, sentences):
        list(map(lambda sentence: self.annotate_sentence(sentence), sentences))
        self.logger.info("Annotated Tokens fastText {}/{}".format(self.coverage, self.total_tokens))
