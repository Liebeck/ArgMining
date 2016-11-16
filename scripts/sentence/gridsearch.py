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

NJOBS = 1
TRAINING_SIZE = 100 # only used in predict.py

def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-nfold', type=int, help='n-fold crossvalidation', default=2)
    argparser.add_argument('-subtask', type=str, required=True, help='Name of the subtask')
    argparser.add_argument('-gridsearchstrategy', type=str, required=True, help='Name of the gridsearch strategy')
    argparser.add_argument('-c', '--classifier', type=str, required=True, help='Name of the classifier')
    argparser.add_argument('--shuffle', type=int, help='Random state of the shuffle or None', default=None)
    return argparser.parse_args()


if __name__ == '__main__':
    t0 = time.time()
    logger = logging.getLogger()
    arguments = config_argparser()
    # 1) Load data sets
    X_train, y_train = load_dataset(file_path='data/THF/sentence/subtask{}_train.json'.format(arguments.subtask))
    # 2) Shuffle if desired

    # 3) Select classifier
    logger.info("Using classifier: {}".format(arguments.classifier))
    classifier, param_grid = get_classifier(arguments.classifier)
    # 4) Select feature combination
    logger.info("Using gridsearch strategy: {}".format(arguments.gridsearchstrategy))
    strategy = GRIDSEARCH_STRATEGIES[arguments.gridsearchstrategy]['features']
    param_grid.update(GRIDSEARCH_STRATEGIES[arguments.gridsearchstrategy]['param_grid'])
    logger.info(param_grid)



    # 5) Start grid search
    pipe = pipeline(strategy=strategy, classifier=classifier)
    logger.info(pipe)
    logger.info("Start grid search")
    gridsearch = GridSearchCV(pipe, param_grid, scoring='f1_macro', cv=arguments.nfold, n_jobs=NJOBS, verbose=2)
    gridsearch.fit(X_train, y_train)
    # 5) Report results
    report_best_results(gridsearch.cv_results_)
    # 6) Serialize the best settings
    settings = OrderedDict()
    settings['classifier'] = arguments.classifier
    settings['gridsearchstrategy'] = arguments.gridsearchstrategy
    settings['subtask'] = arguments.subtask
    settings['nfold'] = arguments.nfold
    settings['shuffle'] = arguments.shuffle
    settings['training_size'] = arguments.shuffle
    best_mean, best_std = best_cv_result(gridsearch.cv_results_)
    settings['gridsearch_best_mean'] = best_mean
    settings['gridsearch_best_std'] = best_std
    settings['gridsearch_parameters'] = gridsearch.best_params_
    if hasattr(classifier, 'random_state'):
        settings['gridsearch_parameters']['classifier__random_state'] = classifier.random_state
    output_path = 'results/sentence/temp/{}_{}_{}'.format(settings['classifier'], settings['strategy'], time.strftime('%Y%m%d_%H%M%S'))
    with open(output_path, 'w') as outfile:
        json.dump(settings, outfile, indent=2)

    logger.info("Total execution time in %0.3fs" % (time.time() - t0))
    logger.info("*****************************************")
