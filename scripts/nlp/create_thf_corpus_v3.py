#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging
import json
import io
from argmining.models.token import Token
from scripts.nlp.spacy_wrapper import SpacyWrapper

logger = logging.getLogger()


def read_annotation_depths(file_path='scripts/nlp/annotation_depths.txt'):
    annotation_depths = {}
    with io.open(file_path, encoding='utf-8') as file:
        content = file.readlines()
        content = [x.strip() for x in content]
        for line in content:
            items = line.split('\t')
            annotation_depths[items[0]] = int(items[1])
    return annotation_depths


def jdefault(o):
    return o.__dict__


def load(file_path='data/THF/sentence/subtaskA_v2_train.json'):
    annotation_depths = read_annotation_depths()
    spacy = SpacyWrapper()
    logger.debug(u'Parsing JSON File: {}'.format(file_path))
    sentences = []
    with open(file_path, encoding='utf-8') as data_file:
        data = json.load(data_file)
        for sentence in data:
            sentence_v3 = {'UniqueID': sentence['UniqueID'],
                           'Label': sentence['Label'],
                           'Text': sentence['Text'],
                           'TextDepth': annotation_depths[sentence['UniqueID'].split('_')[0]],
                           'NLP': spacy.process_sentence(sentence['Text'])}
            sentences.append(sentence_v3)
    logger.info('Parsed {} sentences'.format(len(sentences)))
    output_path = file_path.replace('v2', 'v3')
    logger.info('Saving output to {}'.format(output_path))
    with open(output_path, 'w') as outfile:
        json.dump(sentences, outfile, indent=2, default=jdefault)


def debug_print_sentence(tokens):
    for token in tokens:
        print(
            'tokens.append(Token({}, {}, spacy_pos_universal_google=\'{}\', spacy_ner_type=\'{}\', spacy_ner_iob=\'{}\'))'.format(
                token.token_index_in_sentence,
                token.text.encode('utf-8'),
                token.spacy_pos_universal_google,
                token.spacy_ner_type,
                token.spacy_ner_iob
            ))


if __name__ == '__main__':
    load(file_path='data/THF/sentence/subtaskA_v2_train.json')
    load(file_path='data/THF/sentence/subtaskA_v2_test.json')
    load(file_path='data/THF/sentence/subtaskB_v2_train.json')
    load(file_path='data/THF/sentence/subtaskB_v2_test.json')
