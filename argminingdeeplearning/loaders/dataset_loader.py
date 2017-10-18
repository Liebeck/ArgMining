#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from argminingdeeplearning.models.thf_sentence_deep_learning import THFSentenceDeepLearning
import logging
import json
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
import numpy as np

# Map string labels to integers for Keras
map_class_numeric_A = {'argumentative': 0, 'non-argumentative': 1}
map_class_numeric_B = {'MajorPosition': 0, 'Claim': 1, 'Premise': 2}

logger = logging.getLogger()

def load_dataset(file_path, word_to_index_mapping, subtask, max_length=20):

    logger.debug(u'Parsing JSON File: {}'.format(file_path))
    sentences = []
    with open(file_path, encoding='utf-8') as data_file:
         data = json.load(data_file)
         for sentence in data:
             sentence_tokens = sentence["NLP"]["tokens"]
             label = sentence["Label"]
             if label == 'ClaimContra' or label == 'ClaimPro':
                 label = 'Claim'
             tokens = load_tokens(sentence_tokens, word_to_index_mapping)
             sentence_model = THFSentenceDeepLearning(sentence["UniqueID"], label, sentence["Text"], tokens)
             sentences.append(sentence_model)
    # print(type(sentences))
    # print(type(sentences[0]))
    X = sequence.pad_sequences([item.tokens for item in sentences], maxlen=max_length)
    map_class_numeric = map_class_numeric_A if subtask == 'A' else map_class_numeric_B
    Y_indices = [map_class_numeric[item.label] for item in sentences] # replace string label with index
    Y = to_categorical(np.array(Y_indices), len(set(Y_indices))) # one hot encoded label vector for cross entropy
    unique_ids = [item.uniqueID for item in sentences]
    logger.info('Parsed {} sentences'.format(len(sentences)))
    return X, Y, unique_ids, Y_indices


def load_tokens(sentence_tokens, word_to_index_mapping):
    token_indizes = []
    for token in sentence_tokens:
        word = token['text'].lower()
        token_indizes.append(word_to_index_mapping.get(word, 1))
    return token_indizes


# def load_dataset(file_path, max_length=20, word_to_index_mapping=None, index_to_embedding_mapping=None):
    #if not word_to_index_mapping and not index_to_embedding_mapping:
        # create vocabulary



     # sentence_models = load_sentence_models(file_path=file_path, group_claims=group_claims, text_type=text_type, max_length=max_length)


     #X = dataset
     #y = [item.label for item in dataset]


# def load_sentence_models(file_path='data/THF/sentence/subtaskA_train.json', text_type='lowercase', max_length=20):
#      logger.debug(u'Parsing JSON File: {}'.format(file_path))
#      sentences = []
#      with open(file_path, encoding='utf-8') as data_file:
#          data = json.load(data_file)
#          for sentence in data:
#              sentence_tokens = sentence["NLP"]["tokens"]
#              label = sentence["Label"]
#              if label == 'ClaimContra' or label == 'ClaimPro':
#                  label = 'Claim'
#              tokens = load_tokens(sentence_tokens, text_type, max_length)
#              sentence_model = THFSentenceDeepLearning(sentence["UniqueID"], label, sentence["Text"], tokens)
#              sentences.append(sentence_model)
#      logger.info('Parsed {} sentences'.format(len(sentences)))
#      return sentences

# def load_dataset(file_path, text_type, max_length, data_version='v3', group_claims=True):
#     if data_version == 'v3':
#         dataset = load(file_path=file_path, group_claims=group_claims, text_type=text_type, max_length=max_length)
#     else:
#         raise NotImplementedError
#     X = dataset
#     y = [item.label for item in dataset]
#     return X, y
#
#
# def load_tokens(input_sentence, text_type, max_length):
#     words = []
#     if text_type == 'text':
#         words = list(map(lambda token: token.text, input_sentence.tokens))
#     elif text_type == 'text_lowercase':
#         words = list(map(lambda token: token.text.lower(), input_sentence.tokens))
#     elif text_type.startswith('IWNLP'):
#         lowercase = text_type == 'IWNLP_lowercase'
#         for token in input_sentence.tokens:
#             if token.iwnlp_lemma is not None and len(token.iwnlp_lemma) == 1:
#                 words.append(token.iwnlp_lemma[0] if lowercase else token.iwnlp_lemma[0].lower())
#             else:
#                 words.append(token.text if lowercase else token.text.lower())
#     return words

#
#
# def load(file_path='data/THF/sentence/subtaskA_train.json', text_type='text', max_length=20, group_claims=True):
#     logger.debug(u'Parsing JSON File: {}'.format(file_path))
#     sentences = []
#     with open(file_path, encoding='utf-8') as data_file:
#         data = json.load(data_file)
#         for sentence in data:
#             sentence_tokens = sentence["NLP"]["tokens"]
#             label = sentence["Label"]
#             if group_claims:
#                 if label == 'ClaimContra' or label == 'ClaimPro':
#                     label = 'Claim'
#             tokens = load_tokens(sentence_tokens, text_type, max_length)
#             sentence_model = THFSentenceDeepLearning(sentence["UniqueID"], label, sentence["Text"], tokens)
#             sentences.append(sentence_model)
#     logger.info('Parsed {} sentences'.format(len(sentences)))
#     return sentences
