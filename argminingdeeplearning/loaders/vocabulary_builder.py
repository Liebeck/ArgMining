import logging
import json
import operator
from collections import OrderedDict

logger = logging.getLogger()


def get_word_frequencies(file_path):
    logger.debug(u'Parsing JSON File: {}'.format(file_path))
    word_frequencies = {}
    with open(file_path, encoding='utf-8') as data_file:
        data = json.load(data_file)
        for sentence in data:
            for token in sentence["NLP"]["tokens"]:
                word = token['text'].lower()
                word_frequencies[word] = word_frequencies.get(word, 0) + 1
    logger.info('Parsed {} different words'.format(len(word_frequencies)))
    return word_frequencies


def create_word_to_index_mapping(word_frequencies):
    word_to_index_mapping = {}
    word_to_index_mapping['<PAD>'] = 0  # Padding
    word_to_index_mapping['<OOV>'] = 1  # out of vocabulary
    index = len(word_to_index_mapping)
    sorted_frequencies = OrderedDict(sorted(word_frequencies.items(), key=lambda t: t[1], reverse=True))
    for word, frequency in sorted_frequencies.items():
        word_to_index_mapping[word] = index
        index += 1
    return word_to_index_mapping


def create_index_to_embedding_mapping(word_to_index_mapping, embedding_type='word2vec', dimensionality=100):
    index_to_embedding = {}
    # create empty vector, set to [0]
    # create oov vector random, set to [1]

    for index, word in word_to_index_mapping.items():
        if index == 0 or index == 1:
            continue
        index_to_embedding[index] = None
    return index_to_embedding


def create_mappings(file_path):
    word_frequencies = get_word_frequencies(file_path)
    word_to_index_mapping = create_word_to_index_mapping(word_frequencies)
    index_to_embedding_maping = None
    return word_to_index_mapping, index_to_embedding_maping
