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
from argmining.representations.word2vec import Word2Vec
from argmining.sentence.loaders.THF_sentence_corpus_loader import load_dataset
from argmining.strategies.gridsearch import GRIDSEARCH_STRATEGIES
from argmining.representations.lda import LDA
from argmining.representations.fasttext import FastText

NJOBS = 1


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
                           default='v3')
    argparser.add_argument('-lda_path', type=str, default=None, help='Path to LDA topic model')
    argparser.add_argument('-lda_vocab_path', type=str, default=None, help='Path to LDA vocab')
    argparser.add_argument('-lda_all_words', dest='lda_nouns_only', action='store_false')
    argparser.set_defaults(lda_all_words=False)
    argparser.add_argument('-hilbert', dest='hilbert', action='store_true')
    argparser.set_defaults(hilbert=False)
    argparser.add_argument('-fasttext_path', type=str, default=None, help='Path to fastText model')
    argparser.add_argument('-jobid', type=str, default=None, help='Hilbert job ID')
    return argparser.parse_args()


if __name__ == '__main__':
    t0 = time.time()
    logger = logging.getLogger()
    arguments = config_argparser()
    # 1) Load data sets
    if arguments.data_version == 'v1':
        train_path = 'data/THF/sentence/subtask{}_train.json'.format(arguments.subtask)
    else:
        train_path = 'data/THF/sentence/subtask{}_{}_train.json'.format(arguments.subtask, arguments.data_version)
    if arguments.hilbert:  # work around for absolute paths on the hilbert cluster
        train_path = '/scratch_gs/malie102/jobs/ArgMining/' + train_path
    group_claims = True
    if arguments.subtask == "C":
        group_claims = False
    X_train, y_train = load_dataset(file_path=train_path, data_version=arguments.data_version,
                                    group_claims=group_claims)
    if arguments.embeddings_path:
        word2vec = Word2Vec(model_path=arguments.embeddings_path)
        word2vec.load()
        word2vec.annotate_sentences(X_train)
    if arguments.lda_path:
        lda = LDA(model_path=arguments.lda_path, vocab_path=arguments.lda_vocab_path,
                  nouns_only=arguments.lda_all_words)
        lda.load()
        lda.annotate_sentences(X_train)
    if arguments.fasttext_path:
        fasttext = FastText(model_path=arguments.fasttext_path)
        fasttext.load()
        fasttext.annotate_sentences(X_train)
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
    settings['fasttext_path'] = arguments.fasttext_path
    settings['lda_path'] = arguments.lda_path
    settings['lda_vocab_path'] = arguments.lda_vocab_path
    settings['lda_all_words'] = arguments.lda_all_words
    settings['data_version'] = arguments.data_version
    best_mean, best_std = best_cv_result(gridsearch.cv_results_)
    settings['gridsearch_best_mean'] = best_mean
    settings['gridsearch_best_std'] = best_std
    settings['gridsearch_parameters'] = gridsearch.best_params_
    settings['gridsearch__stratifiedkfold__random_state'] = arguments.gridsearch__stratifiedkfold__random_state
    settings['jobid'] = arguments.jobid
    if hasattr(classifier, 'random_state'):
        settings['gridsearch_parameters']['classifier__random_state'] = classifier.random_state
    if arguments.jobid:
        output_path = 'results/sentence/temp/{}_{}_{}_{}_{}'.format(settings['subtask'], settings['classifier'],
                                                                    settings['gridsearchstrategy'],
                                                                    arguments.jobid,
                                                                    time.strftime('%Y%m%d_%H%M%S'))
    else:
        output_path = 'results/sentence/temp/{}_{}_{}_{}'.format(settings['subtask'], settings['classifier'],
                                                                 settings['gridsearchstrategy'],
                                                                 time.strftime('%Y%m%d_%H%M%S'))
    if arguments.hilbert:  # work around for absolute paths on the hilbert cluster
        output_path = '/scratch_gs/malie102/jobs/ArgMining/' + output_path
    with open(output_path, 'w') as outfile:
        json.dump(settings, outfile, indent=2)

    logger.info("Total execution time in %0.3fs" % (time.time() - t0))
    logger.info("*****************************************")
