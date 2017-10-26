import argparse
import json
from argminingdeeplearning import benchmark
import logging
import sys
import os


def config_logger(log_level=logging.INFO):
    logger = logging.getLogger('')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(log_level)


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining Deep Learning')
    argparser.add_argument('-directory', type=str, required=True, help='Directory of the configuration files')
    return argparser.parse_args()


if __name__ == '__main__':
    config_logger(log_level=logging.INFO)
    logger = logging.getLogger()
    arguments = config_argparser()
    subtasks = ['A', 'B']
    logger.info("Running benchmarks for all files in: {}".format(arguments.directory))
    benchmark_configurations = [f for f in os.listdir(arguments.directory) if f.endswith('.json')]
    logger.info(benchmark_configurations)
    for benchmark_configuration in benchmark_configurations:
        logger.info('Processing: {}'.format(benchmark_configuration))
        path = arguments.directory + benchmark_configuration
        with open(path) as data_file:
            config_parameters = json.load(data_file)
            for subtask in subtasks:
                logger.info('Subtask: {}'.format(subtask))
                logger.info(config_parameters)
                benchmark.benchmark(subtask, config_parameters)
