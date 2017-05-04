import logging
import spacy
import json

class LoaderFlat(object):
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        # self.logger.info('Loading Spacy model')
        # self.nlp = spacy.load('de')
        # self.logger.info('Spacy model loaded')

    def extract_tokens(self, document):
        return None

    def load_thf_corpus_flat(self, path):
        self.logger.info('Loading THF corpus from {}'.format(path))
        with open(path, encoding='utf-8') as data_file:
            data = json.load(data_file)
            proposals = data["instance"]["tempelhofer-feld"]["proposals"]
            tokenized_documents = []
            self.logger.info('Processing {} proposals'.format(len(proposals)))
        # for each proposal
        #   for each comment
        #    recursive for each sub comment

    def flatten_proposal(self, proposal):
        return None

