#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from collections import OrderedDict
import itertools
import copy


def get_default_template(evaluationID, keras_model_name, embeddings_cache_name=None, batch_size=32, epochs=10):
    return {"keras_model_name": keras_model_name,
            "epochs": epochs,
            "embeddings_cache_name": embeddings_cache_name,
            "padding_length": 20,
            "batch_size": batch_size,
            "evaluation_ID": evaluationID,
            "keras_model_parameters": {}
            }


def get_parameter_combinations(dictionary):
    parameter_names = list(dictionary.keys())
    # print(parameter_names)
    parameter_values = list(dictionary.values())
    # print(parameter_values)
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
    meta_parameter_combinations = get_parameter_combinations(meta_parameters)
    model_parameters = OrderedDict([('max_features', [7000]),
                                    ('embedding_size', [128, 300]),
                                    ('dropout', [0.2, 0.4])])
    model_parameter_combinations = get_parameter_combinations(model_parameters)
    parameters = combine_parameters(offset, meta_parameter_combinations, model_parameter_combinations)
    return parameters


def combine_parameters(offset, meta_parameters_combinations, model_parameter_combinations):
    parameters = []
    print('Combining {} meta parameters with {} model parameters'.format(len(meta_parameters_combinations),
                                                                         len(model_parameter_combinations)))
    for meta_parameters in meta_parameters_combinations:
        for model_parameters in model_parameter_combinations:
            parameters_iteration = {}
            parameters_iteration.update(meta_parameters)
            parameters_iteration['evaluation_ID'] = offset
            offset = offset + 1
            parameters_iteration['keras_model_parameters'] = model_parameters
            parameters.append(parameters_iteration)
    print('Returning {} combinations'.format(len(parameters)))
    return parameters



if __name__ == '__main__':
    configs = []
    configs.append(get_lstm_embedding_empty(offset=1))
    print(len(configs))
    for config in configs:
        print(config)
    # todo write all files.
