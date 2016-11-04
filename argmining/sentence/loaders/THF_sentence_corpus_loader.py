from argmining.models.thf_sentence_export import THFSentenceExport
from argmining.models.token import Token
import logging
import xml.etree.ElementTree as ET
import json

logger = logging.getLogger()


def parse_IWNLP_lemma(text):
    return None


def parse_tree_tagger_lemma(text):
    return None


def load(file_path='data/THF/sentence/subtaskA_train.json'):
    """
    Loads the THF corpus from an JSON file
    :param file_path: relative path to the JSON file
    :return:
    """
    logger.debug(u'Parsing JSON File: {}'.format(file_path))

    sentences = []
    with open(file_path, encoding='utf-8') as data_file:
        data = json.load(data_file)
        for sentence in data:
            sentence_tokens = sentence["NLP"]["Sentences"][0]["Tokens"]
            tokens = []
            for token in sentence_tokens:
                token_model = Token(token["TokenIndexInSentence"],
                                    token["Text"],
                                    token["POSTag"],
                                    token["MateToolsPPOS"],
                                    token["MateToolsPLemma"],
                                    parse_tree_tagger_lemma(token["TreeTaggerLemma"]),
                                    parse_IWNLP_lemma(token["IWNLPLemma"]),
                                    token.get("Polarity", None))
                tokens.append(token_model)
            sentence_model = THFSentenceExport(sentence["UniqueID"], sentence["Label"], sentence["Text"], tokens)

            sentences.append(sentence_model)
            # print(json.dumps(token, indent=4, sort_keys=True))
            logger.info('Parsed {} sentences'.format(len(sentences)))
