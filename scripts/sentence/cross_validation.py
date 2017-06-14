import argparse
from argmining.sentence.loaders.THF_sentence_corpus_loader import load_dataset
from time import time
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import logging
from argmining.pipelines.pipeline import pipeline
from argmining.strategies.strategies import STRATEGIES
from argmining.evaluation.gridsearch_report import report_best_results
from argmining.classifiers.classifier import get_classifier
from argmining.evaluation.reduce_training_set import reduce_training_set
from argmining.evaluation.shuffle import shuffle_training_Set
from argmining.representations.word2vec import Word2Vec
from argmining.representations.lda import LDA

NJOBS = 1


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-nfold', type=int, help='n-fold crossvalidation', default=2)
    argparser.add_argument('-subtask', type=str, required=True, help='Name of the subtask')
    argparser.add_argument('-strategy', type=str, required=True, help='Name of the strategy')
    argparser.add_argument('-c', '--classifier', type=str, required=True, help='Name of the classifier')
    argparser.add_argument('--shuffle', type=int, help='Random state of the shuffle or None', default=None)
    argparser.add_argument('--gridsearch__stratifiedkfold__random_state', type=int,
                           help='Random state of the StratifiedKFold for the GridSearchCV', default=123)
    argparser.add_argument('--trainingsize', type=int,
                           help='Amount of training data to be used, e.g. 50 for 50% of the data', default=100)
    argparser.add_argument('--data_version', type=str, help='Version of the data',
                           default='v3')
    #argparser.add_argument('-embeddings', dest='load_embeddings', action='store_true')
    #argparser.set_defaults(load_embeddings=False)
    argparser.add_argument('-embeddings_path', type=str, help='Path to the embeddingsfile', default=None)
    argparser.add_argument('-lda', type=str, default=None, help='Path to LDA topic model')
    argparser.add_argument('-lda_all_words', dest='lda_nouns_only', action='store_false')
    argparser.set_defaults(lda_nouns_only=True)
    return argparser.parse_args()


if __name__ == '__main__':
    t0 = time()
    logger = logging.getLogger()
    arguments = config_argparser()
    # 1) Load data sets
    X_train, y_train = load_dataset(file_path='data/THF/sentence/subtask{}_{}_train.json'.format(arguments.subtask, arguments.data_version),
                                    data_version=arguments.data_version)
    if arguments.embeddings_path:
        word2vec = Word2Vec(model_path=arguments.embeddings_path)
        word2vec.load()
        word2vec.annotate_sentences(X_train)
    if arguments.lda:
        lda = LDA(model_path=arguments.lda, nouns_only=arguments.lda_nouns_only)
        lda.load()
        lda.annotate_sentences(X_train)
    # 2) Shuffle if desired
    X_train, y_train = shuffle_training_Set(X_train, y_train, arguments.shuffle)
    # 4) Reduce training size
    X_train, y_train = reduce_training_set(X_train, y_train, arguments.trainingsize)
    # 5) Select feature combination
    logger.info("Using strategy: {}".format(arguments.strategy))
    strategy = STRATEGIES[arguments.strategy]
    # 6) Select classifier
    logger.info("Using classifier: {}".format(arguments.classifier))
    classifier, param_grid = get_classifier(arguments.classifier, cross_validation=True)
    # 7) Start grid search
    pipe = pipeline(strategy=strategy, classifier=classifier)
    logger.info("Start grid search")
    stratified_k_fold = StratifiedKFold(n_splits=arguments.nfold,
                                        random_state=arguments.gridsearch__stratifiedkfold__random_state)
    gridsearch = GridSearchCV(pipe, param_grid, scoring='f1_macro', cv=stratified_k_fold, n_jobs=NJOBS, verbose=2)
    gridsearch.fit(X_train, y_train)
    # 8) Report results
    report_best_results(gridsearch.cv_results_)
    logger.info("Total execution time in %0.3fs" % (time() - t0))
    logger.info("*****************************************")
