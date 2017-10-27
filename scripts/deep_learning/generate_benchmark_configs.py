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


def get_lstm_embedding_empty(offset=1):
    meta_parameters = OrderedDict([('keras_model_name', ['lstm-embedding-empty']),
                                   ('batch_size', [8, 32]),
                                   ('padding_length', [20]),
                                   ('embeddings_cache_name', [None]),
                                   ('epochs', [10])])
    model_parameters = OrderedDict([('max_features', [7335]),
                                    ('embedding_size', [128, 300]),
                                    ('dropout', [0.2, 0.4])])
    parameters = combine_parameters(offset, meta_parameters, model_parameters)
    return parameters


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
            parameters.append(parameters_iteration)
    return parameters



if __name__ == '__main__':
    configs = []
    configs.extend(get_lstm_embedding_empty(offset=1))
    base_export_path = 'results/sentence_deeplearning/benchmarks'
    for config in configs:
        export_path = '{}/{}_{:03}.json'.format(base_export_path, config['keras_model_name'], config['evaluation_ID'])
        with open(export_path, 'w') as outfile:
            json.dump(config, outfile, indent=2)

