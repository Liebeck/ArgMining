import logging
import json


class IWNLPWrapper(object):
    def __init__(self, lemmatizer_path='data/IWNLP/IWNLP.Lemmatizer_20170501.json'):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug('Loading IWNLP lemmatizer')
        self.load_IWNLP(lemmatizer_path)
        self.logger.debug('IWNLP Lemmatizer loaded')

    def load_IWNLP(self, lemmatizer_path):
        """
        This methods load the IWNLP.Lemmatizer json file and creates a dictionary
         of lowercased forms which maps each form to its possible lemmas.
        """
        self.lemmatizer = {}
        with open(lemmatizer_path, encoding='utf-8') as data_file:
            raw = json.load(data_file)
            for entry in raw:
                self.lemmatizer[entry["Form"]] = entry["Lemmas"]

    def lemmatize(self, word, spacy_pos):
        key = word.lower().trim()
        # IWNLP_pos = self.convert_pos(spacy_pos)
        return None

    def contains_entry(self, word, pos, ignore_case=False):
        if not isinstance(pos, list):
            key = word.lower().strip()
            self.logger.debug(key)
            self.logger.debug(json.dumps(self.lemmatizer[key]))
            if ignore_case:
                return key in self.lemmatizer and any(filter(lambda x: x["POS"] == pos, self.lemmatizer[key]))
            else:
                return key in self.lemmatizer and any(
                    filter(lambda x: x["POS"] == pos and x["Form"] == word, self.lemmatizer[key]))
        else:
            for pos_entry in pos:
                if self.contains_entry(word, pos_entry, ignore_case):
                    return True
            return False


if __name__ == '__main__':
    from argmining.loggers.config import config_logger

    iwnlp_wrapper = IWNLPWrapper()
    logging.debug(iwnlp_wrapper.contains_entry('Hallos', 'Noun', False))
