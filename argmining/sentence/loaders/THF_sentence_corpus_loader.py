#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from argmining.models.thf_sentence_export import THFSentenceExport
from argmining.models.token import Token
from argmining.models.dependency import Dependency
import logging
import json

logger = logging.getLogger()


def load_dataset(file_path, data_version='v2', group_claims=True):
    if data_version == 'v3':
        dataset = load_v3(file_path=file_path, group_claims=group_claims)
    else:
        dataset = load(file_path=file_path, group_claims=group_claims)
    X = dataset
    y = [item.label for item in dataset]
    return X, y


def load_v3(file_path='data/THF/sentence/subtaskA_train.json', group_claims=True):
    logger.debug(u'Parsing JSON File: {}'.format(file_path))
    sentences = []
    with open(file_path, encoding='utf-8') as data_file:
        data = json.load(data_file)
        for sentence in data:
            sentence_tokens = sentence["NLP"]["tokens"]
            tokens = []
            for token in sentence_tokens:
                token.pop("embedding")
                token_model = Token(**token)
                tokens.append(token_model)
            dependencies = []
            dependency_tokens = sentence["NLP"]["dependencies"]
            for dependency in dependency_tokens:
                dependency_model = Dependency(**dependency)
                dependencies.append(dependency_model)
            label = sentence["Label"]
            if group_claims:
                if label == 'ClaimContra' or label == 'ClaimPro':
                    label = 'Claim'
            sentence_model = THFSentenceExport(sentence["UniqueID"], label, sentence["Text"], tokens,
                                               dependencies, textdepth=sentence["TextDepth"])
            sentences.append(sentence_model)
    logger.info('Parsed {} sentences'.format(len(sentences)))
    return sentences


def load(file_path='data/THF/sentence/subtaskA_train.json', group_claims=True):
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
            dependencies = []
            for token in sentence_tokens:
                token_model = Token(token["TokenIndexInSentence"],
                                    token["Text"],
                                    pos_tag=token["POSTag"],
                                    mate_tools_pos_tag=token["MateToolsPPOS"],
                                    mate_tools_lemma=token["MateToolsPLemma"],
                                    tree_tagger_lemma=parse_tree_tagger_lemma(token.get("TreeTaggerLemma", None)),
                                    iwnlp_lemma=parse_IWNLP_lemma(token.get("IWNLPLemma", None)),
                                    polarity=parse_polarity((token.get("Polarity", None))))
                tokens.append(token_model)
            dependency_tokens = sentence["NLP"]["Sentences"][0]["Dependencies"]
            for dependency in dependency_tokens:
                dependency_model = Dependency(dependency["TokenID"], dependency["DependencyRelation"],
                                              dependency["DependencyHeadTokenID"])
                dependencies.append(dependency_model)
            label = sentence["Label"]
            if group_claims:
                if label == 'ClaimContra' or label == 'ClaimPro':
                    label = 'Claim'
            sentence_model = THFSentenceExport(sentence["UniqueID"], label, sentence["Text"], tokens,
                                               dependencies)
            sentences.append(sentence_model)
    logger.info('Parsed {} sentences'.format(len(sentences)))
    return sentences


def parse_IWNLP_lemma(text):
    if not text:
        return None
    else:
        return text


def parse_tree_tagger_lemma(text):
    if not text:
        return None
    else:
        return text


def parse_polarity(polarity):
    return polarity
