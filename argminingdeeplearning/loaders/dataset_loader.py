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
    Y_indices = [map_class_numeric[item.label] for item in sentences]  # replace string label with index
    Y = to_categorical(np.array(Y_indices), len(set(Y_indices)))  # one hot encoded label vector for cross entropy
    unique_ids = [item.uniqueID for item in sentences]
    logger.debug('Parsed {} sentences'.format(len(sentences)))
    return X, Y, unique_ids, Y_indices


def load_tokens(sentence_tokens, word_to_index_mapping):
    token_indizes = []
    for token in sentence_tokens:
        word = token['text'].lower()
        token_indizes.append(word_to_index_mapping.get(word, 1))
    return token_indizes
