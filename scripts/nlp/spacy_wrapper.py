import spacy
import logging
from argmining.models.token import Token
from argmining.models.dependency import Dependency

logger = logging.getLogger()


class SpacyWrapper(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug('Loading Spacy model')
        self.nlp = spacy.load('de')
        self.logger.debug('Spacy model loaded')
        # todo: load sentiws polarity

    def process_sentence(self, sentence):
        result = self.nlp(sentence)
        tokens = []
        dependencies = []
        for token in result:
            token_model = Token(token.i + 1,
                                text=token.text,
                                spacy_pos_stts=token.pos_,
                                spacy_pos_universal_google=token.tag_,
                                spacy_ner_type=token.ent_type_,
                                spacy_ner_iob=token.ent_iob_,
                                spacy_is_punct=token.is_punct,
                                spacy_like_num=token.like_num,
                                spacy_like_url=token.like_url,
                                spacy_shape=token.shape_)
            # Todo: add IWNLP Lemma
            # Todo: Add SentiWS polarity
            tokens.append(token_model)
            dependency_model = Dependency(token.i + 1, token.dep_, token.head.i + 1)
            dependencies.append(dependency_model)
            # print(token_model.token_index_in_sentence, token_model.text.encode('utf-8'), token_model.spacy_pos_stts,
            # token_model.spacy_pos_uts, token_model.spacy_ner_type, token_model.spacy_ner_iob)
            # print(token.i + 1, token.dep_, token.head.i + 1, token.head.text.encode('utf-8'))
        return {
            'tokens': tokens,
            'dependencies': dependencies
        }


if __name__ == '__main__':
    spacy_wrapper = SpacyWrapper()
    spacy_wrapper.process_sentence('Das ist ein schöner Testsatz mit schlechter Bewertung.')
