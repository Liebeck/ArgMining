import argparse
from argmining.sentence.loaders.THF_sentence_corpus_loader import load
import time
from sklearn.model_selection import GridSearchCV
import logging
import json
from argmining.pipelines.pipeline import pipeline
from argmining.strategies.strategies import STRATEGIES
from argmining.evaluation.gridsearch_report import report
from argmining.classifiers.classifier import get_classifier
from collections import OrderedDict

NJOBS = 1


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-configfile', type=str, required=True, help='Name of the subtask')
    return argparser.parse_args()


if __name__ == '__main__':
    t0 = time.time()
    logger = logging.getLogger()
    arguments = config_argparser()
    # 1) Read settings file

    # 2) Read datasets



    logger.info("Total execution time in %0.3fs" % (time.time() - t0))
    logger.info("*****************************************")
