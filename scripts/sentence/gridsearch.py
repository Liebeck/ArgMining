import argparse
from argmining.sentence.loaders.THF_sentence_corpus_loader import load_dataset
import time
from sklearn.model_selection import GridSearchCV
import logging
import json
from argmining.pipelines.pipeline import pipeline
from argmining.strategies.gridsearch import GRIDSEARCH_STRATEGIES
from argmining.evaluation.gridsearch_report import report_best_results, best_cv_result
from argmining.classifiers.classifier import get_classifier
from collections import OrderedDict
import copy
from argmining.evaluation.reduce_training_set import reduce_training_set

NJOBS = 1
TRAINING_SIZE = 100  # only used in predict.py


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-nfold', type=int, help='n-fold crossvalidation', default=2)
    argparser.add_argument('-subtask', type=str, required=True, help='Name of the subtask')
    argparser.add_argument('-gridsearchstrategy', type=str, required=True, help='Name of the gridsearch strategy')
    argparser.add_argument('-c', '--classifier', type=str, required=True, help='Name of the classifier')
    argparser.add_argument('--shuffle', type=int, help='Random state of the shuffle or None', default=None)
    argparser.add_argument('--trainingsize', type=int,
                           help='Amount of training data to be used, e.g. 50 for 50% of the data', default=100)
    return argparser.parse_args()


if __name__ == '__main__':
    t0 = time.time()
    logger = logging.getLogger()
    arguments = config_argparser()
    # 1) Load data sets
    X_train, y_train = load_dataset(file_path='data/THF/sentence/subtask{}_train.json'.format(arguments.subtask))
    # 2) Shuffle if desired

    # 3) Reduce training size
    X_train, y_train = reduce_training_set(X_train, y_train, arguments.trainingsize)
    # 4) Select classifier
    logger.info("Using classifier: {}".format(arguments.classifier))
    classifier, param_grid_clf = get_classifier(arguments.classifier)
    # 5) Select feature combination
    logger.info("Using gridsearch strategy: {}".format(arguments.gridsearchstrategy))
    strategy = GRIDSEARCH_STRATEGIES[arguments.gridsearchstrategy]['features']
    strategy_built = []
    for feature_name, feature in strategy.items():
        strategy_built.append(feature())
    param_grid = []
    if type(GRIDSEARCH_STRATEGIES[arguments.gridsearchstrategy]['param_grid']) is list:
        for dict in GRIDSEARCH_STRATEGIES[arguments.gridsearchstrategy]['param_grid']:
            print(dict)
            new_dict = copy.deepcopy(dict)
            new_dict.update(param_grid_clf)
            param_grid.append(new_dict)
    else:
        param_grid = GRIDSEARCH_STRATEGIES[arguments.gridsearchstrategy]['param_grid']
        param_grid.update(param_grid_clf)
    logger.info(param_grid)
    # 6) Start grid search
    pipe = pipeline(strategy=strategy_built, classifier=classifier)
    logger.info(pipe)
    logger.info("Start grid search")
    gridsearch = GridSearchCV(pipe, param_grid, scoring='f1_macro', cv=arguments.nfold, n_jobs=NJOBS, verbose=2)
    gridsearch.fit(X_train, y_train)
    # 7) Report results
    report_best_results(gridsearch.cv_results_)
    # 8) Serialize the best settings
    settings = OrderedDict()
    settings['classifier'] = arguments.classifier
    settings['gridsearchstrategy'] = arguments.gridsearchstrategy
    settings['subtask'] = arguments.subtask
    settings['nfold'] = arguments.nfold
    settings['shuffle'] = arguments.shuffle
    settings['training_size'] = arguments.trainingsize
    best_mean, best_std = best_cv_result(gridsearch.cv_results_)
    settings['gridsearch_best_mean'] = best_mean
    settings['gridsearch_best_std'] = best_std
    settings['gridsearch_parameters'] = gridsearch.best_params_
    if hasattr(classifier, 'random_state'):
        settings['gridsearch_parameters']['classifier__random_state'] = classifier.random_state
    output_path = 'results/sentence/temp/{}_{}_{}'.format(settings['classifier'], settings['gridsearchstrategy'],
                                                          time.strftime('%Y%m%d_%H%M%S'))
    with open(output_path, 'w') as outfile:
        json.dump(settings, outfile, indent=2)

    logger.info("Total execution time in %0.3fs" % (time.time() - t0))
    logger.info("*****************************************")
