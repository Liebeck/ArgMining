#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging
import json
import io
from argmining.models.token import Token

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


def load(file_path='data/THF/sentence/subtaskA_v2_train.json'):
    annotation_depths = read_annotation_depths()
    logger.debug(u'Parsing JSON File: {}'.format(file_path))
    sentences = []
    with open(file_path, encoding='utf-8') as data_file:
        data = json.load(data_file)
        for sentence in data:
            sentence_v3 = {'UniqueID': sentence['UniqueID'],
                           'Label': sentence['Label'],
                           #'Text': sentence['Text'],
                           'TextDepth': annotation_depths[sentence['UniqueID'].split('_')[0]],
                           'NLP': None}
            sentences.append(sentence_v3)
            #print(sentence_v3)


            # }
            # sentence_tokens = sentence["NLP"]["Sentences"][0]["Tokens"]
            # tokens = []
            # dependencies = []
            # for token in sentence_tokens:
            #     token_model = Token(token["TokenIndexInSentence"],
            #                         token["Text"],
            #                         pos_tag=token["POSTag"],
            #                         mate_tools_pos_tag=token["MateToolsPPOS"],
            #                         mate_tools_lemma=token["MateToolsPLemma"],
            #                         tree_tagger_lemma=parse_tree_tagger_lemma(token.get("TreeTaggerLemma", None)),
            #                         iwnlp_lemma=parse_IWNLP_lemma(token.get("IWNLPLemma", None)),
            #                         polarity=parse_polarity((token.get("Polarity", None))))
            #     tokens.append(token_model)
            # dependency_tokens = sentence["NLP"]["Sentences"][0]["Dependencies"]
            # for dependency in dependency_tokens:
            #     dependency_model = Dependency(dependency["TokenID"], dependency["DependencyRelation"],
            #                                   dependency["DependencyHeadTokenID"])
            #     dependencies.append(dependency_model)
            # label = sentence["Label"]
            # if group_claims:
            #     if label == 'ClaimContra' or label == 'ClaimPro':
            #         label = 'Claim'
            # sentence_model = THFSentenceExport(sentence["UniqueID"], label, sentence["Text"], tokens,
            #                                    dependencies)
            # sentences.append(sentence_model)
    logger.info('Parsed {} sentences'.format(len(sentences)))
    output_path = file_path.replace('v2', 'v3')
    logger.info('Saving output to {}'.format(output_path))
    with open(output_path, 'w') as outfile:
        json.dump(sentences, outfile, indent=2)


if __name__ == '__main__':
    load(file_path='data/THF/sentence/subtaskA_v2_train.json')
