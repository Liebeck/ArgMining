import logging
import json
from collections import OrderedDict
import numpy as np
from collections import Counter

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


def create_index_to_embedding_mapping(word_to_index_mapping, word_to_embedding_cache):
    if word_to_embedding_cache is None:
        return None
    index_to_embedding_mapping = {}
    embedding_length = len(word_to_embedding_cache[next(iter(word_to_index_mapping))])
    index_to_embedding_mapping[0] = [0.0] * embedding_length  # create empty vector as the padding vector, set to [0]
    oov_vector = np.random.rand(embedding_length)  # create oov vector random, set to [1]
    # print(oov_vector)
    index_to_embedding_mapping[1] = oov_vector
    # print('sport' in word_to_embedding_cache)
    # print(word_to_embedding_cache['sport'])
    for word, index in word_to_index_mapping.items():
        if index == 0 or index == 1:
            continue
        # if word in word_to_embedding_cache:
        if word_to_embedding_cache[word] is None:
            index_to_embedding_mapping[index] = oov_vector
        else:
            index_to_embedding_mapping[index] = word_to_embedding_cache[word]
    return index_to_embedding_mapping


def create_mappings(train_path, test_path, word_to_embedding_cache):
    word_frequencies_train = Counter(get_word_frequencies(train_path))
    word_frequencies_test = Counter(get_word_frequencies(test_path))
    word_frequencies = word_frequencies_train + word_frequencies_test
    # print(word_frequencies)
    word_to_index_mapping = create_word_to_index_mapping(word_frequencies)
    index_to_embedding_maping = create_index_to_embedding_mapping(word_to_index_mapping, word_to_embedding_cache)
    return word_to_index_mapping, index_to_embedding_maping
