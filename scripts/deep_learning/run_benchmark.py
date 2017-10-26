import argparse
import json
from argminingdeeplearning import benchmark
import logging
import sys


def config_logger(log_level=logging.INFO):
    logger = logging.getLogger('')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(log_level)


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining Deep Learning')
    argparser.add_argument('-configpath', type=str, required=True, help='Path to the configuration file')
    return argparser.parse_args()


if __name__ == '__main__':
    config_logger(log_level=logging.INFO)
    logger = logging.getLogger()
    arguments = config_argparser()
    logger.info("Running benchmarkfile: {}".format(arguments.configpath))
    with open(arguments.configpath) as data_file:
        config_parameters = json.load(data_file)
        logger.info(config_parameters)
        benchmark.benchmark(config_parameters)
