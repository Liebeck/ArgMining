import argparse
from argmining.sentence.loaders.THF_sentence_corpus_loader import load_dataset
import time
import logging
import json
from argmining.pipelines.pipeline import pipeline
from argmining.strategies.gridsearch import GRIDSEARCH_STRATEGIES
from argmining.classifiers.classifier import create_classifier

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
    logger.info("Loading config file: {}".format(arguments.configfile))
    with open(arguments.configfile) as data_file:
        settings = json.load(data_file)
    # 2) Read datasets
    X_train, y_train = load_dataset(file_path='data/THF/sentence/subtask{}_train.json'.format(settings['subtask']))
    X_test, y_test = load_dataset(file_path='data/THF/sentence/subtask{}_test.json'.format(settings['subtask']))
    # 3) Load classifier with arguments
    classifer_parameters = {}
    for key, value in settings['gridsearch_parameters'].items():
        if key.startswith('classifier__'):
            classifer_parameters[key.replace('classifier__', '')] = value
    logger.info('Classifier arguments: {}'.format(classifer_parameters))
    logger.info('Creating classifier {}'.format(settings['classifier']))
    classifier = create_classifier(settings['classifier'], classifer_parameters)
    logger.info(classifier)
    # 4) Load features and set arguments
    logger.info("Using gridsearch strategy: {}".format(settings['gridsearchstrategy']))
    strategy = GRIDSEARCH_STRATEGIES[settings['gridsearchstrategy']]['features']
    strategy_built = []
    for feature_name, feature in strategy.items():
        feature_parameters = {}
        for key, value in settings['gridsearch_parameters'].items():
            if key.startswith('union__{}'.format(feature_name)):
                feature_parameters[key.replace('union__{}__transformer__'.format(feature_name), '')] = value
        if not feature_parameters:
            logger.info(
                'Building feature {} without parameters'.format(feature_name))
            strategy_built.append(feature.build())
        else:
            logger.info(
                'Building feature {} with the following parameters: {}'.format(feature_name, feature_parameters))
            strategy_built.append(feature.build(**feature_parameters))
    # 5) Train classifier
    # 6) Predict the test set
    # 7) Print the confusion matrix
    # 8) Save the prediction into the file system

    logger.info("Total execution time in %0.3fs" % (time.time() - t0))
    logger.info("*****************************************")
