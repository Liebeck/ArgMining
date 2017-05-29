import spacy
import logging
from argmining.models.token import Token
from argmining.models.dependency import Dependency
from iwnlp.iwnlp_wrapper import IWNLPWrapper
from scripts.nlp.sentiws_wrapper import SentiWSWrapper

logger = logging.getLogger()


def format_iwnlp_lemma(input):
    if input is None:
        return ''
    else:
        return [lemma.encode('utf-8') for lemma in input]


class SpacyWrapper(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.lemmatizer = IWNLPWrapper(lemmatizer_path='data/IWNLP/IWNLP.Lemmatizer_20170501.json')
        self.sentiws = SentiWSWrapper(sentiws_path='data/sentiws')
        self.logger.debug('Loading Spacy model')
        self.nlp = spacy.load('de')
        self.logger.debug('Spacy model loaded')

        # todo: load sentiws polarity

    def process_sentence(self, sentence):
        result = self.nlp(sentence)
        tokens = []
        dependencies = []
        for token in result:
            iwnlp_lemma = self.lemmatizer.lemmatize(token.text, pos_universal_google=token.pos_)
            sentiws = self.sentiws.determine(token.text)
            token_model = Token(token.i + 1,
                                text=token.text,
                                spacy_pos_stts=token.pos_,
                                spacy_pos_universal_google=token.tag_,
                                iwnlp_lemma=iwnlp_lemma,
                                spacy_ner_type=token.ent_type_,
                                spacy_ner_iob=token.ent_iob_,
                                spacy_is_punct=token.is_punct,
                                spacy_like_num=token.like_num,
                                spacy_like_url=token.like_url,
                                spacy_shape=token.shape_,
                                polarity_sentiws=sentiws)
            # Todo: Add SentiWS polarity
            tokens.append(token_model)
            dependency_model = Dependency(token.i + 1, token.dep_, token.head.i + 1)
            dependencies.append(dependency_model)
            print(token_model.token_index_in_sentence, token_model.text.encode('utf-8'),
                  format_iwnlp_lemma(token_model.iwnlp_lemma), token_model.spacy_pos_stts,
                  token_model.spacy_pos_universal_google, token_model.spacy_ner_type, token_model.spacy_ner_iob)
        return {
            'tokens': tokens,
            'dependencies': dependencies
        }


if __name__ == '__main__':
    spacy_wrapper = SpacyWrapper()
    spacy_wrapper.process_sentence('Das ist ein guter, schöner Testsatz mit schlechter Bewertung.')
