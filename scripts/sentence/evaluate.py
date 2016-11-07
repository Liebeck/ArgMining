import argparse
from argmining.sentence.loaders.THF_sentence_corpus_loader import load
from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import logging
from argmining.pipelines.pipeline import pipeline
from argmining.strategies.strategies import STRATEGIES

NJOBS = 1


def config_argparser():
    argparser = argparse.ArgumentParser(description='ArgMining')
    argparser.add_argument('-nfold', type=int, help='n-fold crossvalidation', default=10)
    argparser.add_argument('-subtask', type=str, required=True, help='Name of the subtask')
    argparser.add_argument('-strategy', type=str, required=True, help='Name of the strategy')
    argparser.add_argument('-c', '--classifier', type=str, required=True, help='Name of the classifier')
    argparser.add_argument('--shuffle', type=int, help='Random state of the shuffle or None', default=None)
    return argparser.parse_args()





if __name__ == '__main__':
    t0 = time()
    logger = logging.getLogger()
    arguments = config_argparser()
    # 1) Load data sets
    dataset = load(file_path='data/THF/sentence/subtask{}_train.json'.format(arguments.subtask))

    X_train = dataset
    y_train = [item.label for item in dataset]
    # 2) Shuffle if desired
    # 3)
    # param_grid = {'classifier__C': [1e3, 5e3, 1e4, 5e4, 1e5],
    # 'classifier__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
    param_grid = {'classifier__C': [1e3],
                  'classifier__gamma': [0.005, 0.01]}
    classifier = SVC(kernel='rbf', class_weight='balanced', random_state=0)
    logger.info("Using strategy: {}".format(arguments.strategy))
    strategy = STRATEGIES[arguments.strategy]
    pipe = pipeline(strategy=strategy, classifier=classifier)
    logger.info("Start pipeline fit")
    pipe.fit(X_train, y_train)
    logger.info("Start grid search")
    gridsearch = GridSearchCV(pipe, param_grid, scoring='f1_macro', cv=arguments.nfold, n_jobs=NJOBS, verbose=2)
    gridsearch.fit(X_train, y_train)
    report(gridsearch.grid_scores_)
    logger.info("Total execution time in %0.3fs" % (time() - t0))
    logger.info("*****************************************")
