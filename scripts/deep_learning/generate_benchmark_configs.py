#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from collections import OrderedDict
import itertools
import json


def get_parameter_combinations(dictionary):
    parameter_names = list(dictionary.keys())
    parameter_values = list(dictionary.values())
    unnamed_combinations = list(itertools.product(*parameter_values))
    combination_parameters = []
    for combination in unnamed_combinations:
        combination_parameter = {}
        for index in range(len(combination)):
            combination_parameter[parameter_names[index]] = combination[index]
        combination_parameters.append(combination_parameter)
    return combination_parameters


def combine_parameters(offset, meta_parameters, model_parameters):
    meta_parameter_combinations = get_parameter_combinations(meta_parameters)
    model_parameter_combinations = get_parameter_combinations(model_parameters)
    parameters = []
    print('Combining {} meta parameters with {} model parameters'.format(len(meta_parameter_combinations),
                                                                         len(model_parameter_combinations)))
    for meta_parameters in meta_parameter_combinations:
        for model_parameters in model_parameter_combinations:
            parameters_iteration = {}
            parameters_iteration.update(meta_parameters)
            parameters_iteration['evaluation_ID'] = offset
            offset = offset + 1
            parameters_iteration['keras_model_parameters'] = model_parameters
            parameters.append(OrderedDict(parameters_iteration))
    return parameters


def generate_execution_script(configs):
    with open('scripts/deep_learning/all_benchmarks.sh', 'w') as outfile:
        outfile.write("# !/usr/bin/env bash\n")
        outfile.write("export PYTHONPATH=$PYTHONPATH:/home/matthias/Documents/ArgMining\n\n")
        for config in configs:
            for subtask in ['A', 'B']:
                configpath = 'results/sentence_deeplearning/benchmarks/{}_{:03}.json'.format(config['keras_model_name'],
                                                                                             config['evaluation_ID'])
                outfile.write(
                    'python3 scripts/deep_learning/run_single_benchmark.py -subtask {} -configpath {}\n'.format(subtask,
                                                                                                                configpath))


def get_lstm_embedding_empty(offset):
    meta_parameters = OrderedDict([('keras_model_name', ['lstm-embedding-empty']),
                                   ('batch_size', [32]),
                                   ('padding_length', [20]),
                                   ('embeddings_cache_name', [None]),
                                   ('epochs', [10])])
    model_parameters = OrderedDict([('max_features', [7335]),
                                    ('embedding_size', [128, 300]),
                                    ('dropout', [0.2, 0.4])])
    parameters = combine_parameters(offset, meta_parameters, model_parameters)
    return parameters


def get_lstm_embedding_pretrained(offset):
    meta_parameters = OrderedDict([('keras_model_name', ['lstm-embedding-pretrained']),
                                   ('batch_size', [32]),
                                   ('padding_length', [20]),
                                   ('embeddings_cache_name', ['word2vec_wiki_de_20170501_300-reduced-both']),
                                   ('epochs', [10, 20])])
    model_parameters = OrderedDict([('lstm_size', [128, 64]),
                                    ('padding_length', [20]),
                                    ('dropout', [0.2, 0.4, 0.8, 0.9])])
    parameters = combine_parameters(offset, meta_parameters, model_parameters)
    return parameters

def get_lstm_stacked(offset, keras_model_name='lstm-stacked'):
    meta_parameters = OrderedDict([('keras_model_name', [keras_model_name]),
                                   ('batch_size', [32]),
                                   ('padding_length', [20]),
                                   ('embeddings_cache_name', ['word2vec_wiki_de_20170501_300-reduced-both']),
                                   ('epochs', [10, 20])])
    model_parameters = OrderedDict([('lstm_size_layer1', [128]),
                                    ('lstm_size_layer2', [128]),
                                    ('padding_length', [20]),
                                    ('dropout', [0.2, 0.5, 0.7, 0.8, 0.85, 0.9])])
    parameters = combine_parameters(offset, meta_parameters, model_parameters)
    offset = offset + len(parameters)

    model_parameters = OrderedDict([('lstm_size_layer1', [128]),
                                    ('lstm_size_layer2', [64]),
                                    ('padding_length', [20]),
                                    ('dropout', [0.2, 0.5, 0.7, 0.8, 0.85, 0.9])])
    more_parameters = combine_parameters(offset, meta_parameters, model_parameters)
    parameters.extend(more_parameters)
    offset = offset + len(more_parameters)
    model_parameters = OrderedDict([('lstm_size_layer1', [64]),
                                    ('lstm_size_layer2', [32]),
                                    ('padding_length', [20]),
                                    ('dropout', [0.2, 0.5, 0.7, 0.8, 0.85, 0.9])])
    parameters.extend(combine_parameters(offset, meta_parameters, model_parameters))
    return parameters


def get_embedding_cnn(offset):
    meta_parameters = OrderedDict([('keras_model_name', ['embedding_cnn']),
                                   ('batch_size', [32]),
                                   ('padding_length', [20]),
                                   ('embeddings_cache_name', ['word2vec_wiki_de_20170501_300-reduced-both']),
                                   ('epochs', [10, 20])])
    model_parameters = OrderedDict([('filters', [250, 175, 100]),
                                    ('padding_length', [20]),
                                    ('kernel_size', [3, 5]),
                                    ('dropout', [0.2, 0.5, 0.7, 0.8, 0.9])])
    parameters = combine_parameters(offset, meta_parameters, model_parameters)
    return parameters


def get_embedding_cnn_lstm(offset):
    meta_parameters = OrderedDict([('keras_model_name', ['embedding_cnn_lstm']),
                                   ('batch_size', [32]),
                                   ('padding_length', [20]),
                                   ('embeddings_cache_name', ['word2vec_wiki_de_20170501_300-reduced-both']),
                                   ('epochs', [10, 20])])
    model_parameters = OrderedDict([('filters', [250, 175, 100]),
                                    ('padding_length', [20]),
                                    ('kernel_size', [3, 5]),
                                    ('pool_size', [4]),
                                    ('lstm_size_layer', [70, 128]),
                                    ('dropout', [0.2, 0.5, 0.7, 0.9])])
    parameters = combine_parameters(offset, meta_parameters, model_parameters)
    return parameters


if __name__ == '__main__':
    configs = []
    configs.extend(get_lstm_embedding_empty(offset=1))
    configs.extend(get_lstm_embedding_pretrained(offset=100))
    configs.extend(get_lstm_stacked(offset=200))
    configs.extend(get_lstm_stacked(offset=300, keras_model_name='blstm'))
    configs.extend(get_embedding_cnn(offset=400))
    configs.extend(get_embedding_cnn_lstm(offset=500))
    base_export_path = 'results/sentence_deeplearning/benchmarks'
    for config in configs:
        export_path = '{}/{}_{:03}.json'.format(base_export_path, config['keras_model_name'], config['evaluation_ID'])
        with open(export_path, 'w') as outfile:
            json.dump(config, outfile, indent=2)
    generate_execution_script(configs)
