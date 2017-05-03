import argparse
import copy
import json
import logging
import time
from collections import OrderedDict
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from argmining.classifiers.classifier import get_classifier
from argmining.evaluation.gridsearch_report import report_best_results, best_cv_result
from argmining.evaluation.reduce_training_set import reduce_training_set
from argmining.evaluation.shuffle import shuffle_training_Set
from argmining.pipelines.pipeline import pipeline
from argmining.resources.word2vec import Word2Vec
from argmining.sentence.loaders.THF_sentence_corpus_loader import load_dataset
from argmining.strategies.gridsearch import GRIDSEARCH_STRATEGIES

NJOBS = 1
TRAINING_SIZE = 100  # only used in predict.py


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-nfold', type=int, help='n-fold crossvalidation', default=2)
    argparser.add_argument('-subtask', type=str, required=True, help='Name of the subtask')
    argparser.add_argument('-gridsearchstrategy', type=str, required=True, help='Name of the gridsearch strategy')
    argparser.add_argument('-c', '--classifier', type=str, required=True, help='Name of the classifier')
    argparser.add_argument('--shuffle', type=int, help='Random state of the shuffle or None', default=None)
    argparser.add_argument('--gridsearch__stratifiedkfold__random_state', type=int,
                           help='Random state of the StratifiedKFold for the GridSearchCV', default=123)
    argparser.add_argument('--trainingsize', type=int,
                           help='Amount of training data to be used, e.g. 50 for 50% of the data', default=100)
    argparser.add_argument('-embeddings_path', type=str, help='Path to the embeddingsfile', default=None)
    argparser.add_argument('--data_version', type=str, help='Version of the data',
                           default='v1')
    argparser.add_argument('-hilbert', dest='hilbert', action='store_true')
    argparser.set_defaults(hilbert=False)
    return argparser.parse_args()


if __name__ == '__main__':
    t0 = time.time()
    logger = logging.getLogger()
    arguments = config_argparser()
    # 1) Load data sets
    if arguments.data_version == 'v1':
        train_path = 'data/THF/sentence/subtask{}_train.json'.format(arguments.subtask)
    elif arguments.data_version == 'v2':
        train_path = 'data/THF/sentence/subtask{}_v2_train.json'.format(arguments.subtask)
    if arguments.hilbert:  # work around for absolute paths on the hilbert cluster
        train_path = '/home/malie102/jobs/ArgMining/' + train_path
    X_train, y_train = load_dataset(file_path=train_path)
    if arguments.embeddings_path:
        word2vec = Word2Vec(model_path=arguments.embeddings_path)
        word2vec.load()
        word2vec.annotate_sentences(X_train)
    # 2) Shuffle if desired
    X_train, y_train = shuffle_training_Set(X_train, y_train, arguments.shuffle)
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
    stratified_k_fold = StratifiedKFold(n_splits=arguments.nfold,
                                        random_state=arguments.gridsearch__stratifiedkfold__random_state)
    gridsearch = GridSearchCV(pipe, param_grid, scoring='f1_macro', cv=stratified_k_fold, n_jobs=NJOBS, verbose=2)
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
    settings['embeddings_path'] = arguments.embeddings_path
    settings['data_version'] = arguments.data_version
    best_mean, best_std = best_cv_result(gridsearch.cv_results_)
    settings['gridsearch_best_mean'] = best_mean
    settings['gridsearch_best_std'] = best_std
    settings['gridsearch_parameters'] = gridsearch.best_params_
    if hasattr(classifier, 'random_state'):
        settings['gridsearch_parameters']['classifier__random_state'] = classifier.random_state
    output_path = 'results/sentence/temp/{}_{}_{}_{}'.format(settings['subtask'], settings['classifier'],
                                                             settings['gridsearchstrategy'],
                                                             time.strftime('%Y%m%d_%H%M%S'))
    if arguments.hilbert:  # work around for absolute paths on the hilbert cluster
        output_path = '/home/malie102/jobs/ArgMining/' + output_path
    with open(output_path, 'w') as outfile:
        json.dump(settings, outfile, indent=2)

    logger.info("Total execution time in %0.3fs" % (time.time() - t0))
    logger.info("*****************************************")
